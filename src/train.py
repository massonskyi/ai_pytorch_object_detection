import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ai_pytorch_object_detection.model.v4Model import v4Model
from ai_pytorch_object_detection.src.v4Dataset import v4Dataset


def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 99:  # Print the loss every 100 batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
                running_loss = 0.0

# yaml_path = "/home/user064/repo/pyrepo/ai_test/ai_pytorch_object_detection/model/yolov8n.yaml"
# model = v4Model(yaml_path)

# # Assuming you have the paths to your image and label directories
# image_dir = '/home/user064/repo/pyrepo/ai_test/yolo_model/datasets_v1_1000ph/train/images'
# label_dir = '/home/user064/repo/pyrepo/ai_test/yolo_model/datasets_v1_1000ph/train/labels'
#
# train_data = v4Dataset(image_dir, label_dir)
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
#
# # Define your loss function and optimizer
# criterion = nn.MSELoss()  # Or any other loss function that suits your task
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Or any other optimizer that suits your task
#
# # Train the model
# num_epochs = 10  # Or any other number that suits your task
# train(model, train_loader, criterion, optimizer, num_epochs)


# model.train(data="/home/user064/repo/pyrepo/ai_test/yolo_model/datasets_v1_1000ph/data.yaml")