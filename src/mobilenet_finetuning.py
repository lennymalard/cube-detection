import os
import torch, torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.datasets import CocoDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CUSTOM_CLASSES = ["black box", "blue box", "green box", "red box", "white box"]
NUM_CLASSES = len(CUSTOM_CLASSES) + 1

weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
model = ssdlite320_mobilenet_v3_large(weights=weights)

old_cls_head = model.head.classification_head.module_list
in_channels = []
for layer in old_cls_head:
    in_channels.append(layer[-1].in_channels)

num_anchors = model.anchor_generator.num_anchors_per_location()

new_cls_head = SSDLiteClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=NUM_CLASSES,
    norm_layer=nn.BatchNorm2d,
)

model.head.classification_head = new_cls_head

def transforms(input, target):
    if len(target) == 0:
        return None, None
    old_size = input.size
    input = Image.fromarray(cv2.resize(np.array(input), (320, 320)))
    x_factor = input.size[0] / old_size[0]
    y_factor = input.size[1] / old_size[1]
    category_id_list = []
    bbox_list = []
    for i in range(len(target)):
        bbox_list.append(
            [
                target[i]['bbox'][0] * x_factor,
                target[i]['bbox'][1] * y_factor,
                target[i]['bbox'][0] * x_factor + target[i]['bbox'][2] * x_factor,
                target[i]['bbox'][1] * y_factor + target[i]['bbox'][3] * y_factor
            ]
        )
        category_id_list.append(target[i]['category_id'])
    new_target = {}
    new_target['labels'] = torch.tensor(category_id_list)
    new_target['boxes'] = torch.tensor(bbox_list)
    return weights.transforms()(input), new_target

training_dataset = CocoDetection(
    root='../bouling box.v5i.coco/train',
    annFile='../data/bouling box.v5i.coco/train/_annotations.coco.json',
    transforms=transforms
)

eval_dataset = CocoDetection(
    root='../data/bouling box.v5i.coco/valid',
    annFile='../data/bouling box.v5i.coco/valid/_annotations.coco.json',
    transforms=transforms
)

def collate_fn(batch):
    batch = [sample for sample in batch if sample != (None, None)]
    inputs = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    return torch.stack(inputs, dim=0), targets

training_dataloader = DataLoader(dataset=training_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

def training_loop(model, optimizer, dataloader, epochs):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets_on_gpu = []
            for target in targets:
                targets_on_gpu.append({k: v.to(device) for k, v in target.items()})
            loss_dict = model(inputs, targets_on_gpu)
            loss = torch.add(loss_dict['bbox_regression'], loss_dict['classification'])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")
    return model.state_dict()

adam = torch.optim.Adam(model.parameters())
parameters = training_loop(model, adam, training_dataloader, 25)

eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

def evaluation_loop(model, dataloader):
    mean_ap = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    mean_ap.to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets_on_gpu = []
            for target in targets:
                targets_on_gpu.append({k: v.to(device) for k, v in target.items()})
            predictions = model(inputs)
            mean_ap.update(predictions, targets_on_gpu)
    final_mean_ap = mean_ap.compute()
    print(f"Final Mean AP: {final_mean_ap}")

evaluation_loop(model, eval_dataloader)

torch.save(parameters, './parameters.pt')
torch.save(model, './model.pt')