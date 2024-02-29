import os
import torch

BATCH_SIZE = 4  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 25  # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASSES = [
    'FPV_DRONE'
]
NUM_CLASSES = 2
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

DATASET_DIR = f"{os.getcwd()}/dataset"

# training images and XML files directory
TRAIN_DIR = f'{DATASET_DIR}/train'
# validation images and XML files directory
VALID_DIR = f'{DATASET_DIR}/valid'

# location to save model and plots
OUT_DIR = './outputs'

SAVE_PLOTS_EPOCH = 2  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2  # save model after these many epochs


MODEL_YAML = f"{os.getcwd()}/model/model.yaml"