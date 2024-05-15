from copy import deepcopy
from pathlib import Path

import torch
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import yaml_load, ROOT
from ultralytics.utils.torch_utils import initialize_weights, scale_img

from ai_pytorch_object_detection.model.BaseModel import BaseModel
from ai_pytorch_object_detection.src.nn import LOGGER
from ai_pytorch_object_detection.src.nn.loss import v4DetectionLoss
from ai_pytorch_object_detection.src.nn.modules.block import Detect, yaml_model_load, parse_model


class V4DetectModel(Model):
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = V4World(path)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionV4Model,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }


class DetectionV4Model(BaseModel):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v4DetectionLoss(self)


class V4World(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt") -> None:
        super().__init__(model=model, task="detect")
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets_v1_1000ph/coco8.yaml").get("names")

    @property
    def task_map(self):
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
