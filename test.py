from ultralytics import YOLO


model = YOLO("./outputs/model2.pt")

model(source="/home/user064/repo/pyrepo/ai/ai_pytorch_object_detection/data/test/images/64dde2f302e8bd0cb20df32a_jpg"
             ".rf.795a24dd4e7cf5cdeeed46969df6ff4b.jpg", show=True, conf=0.4)