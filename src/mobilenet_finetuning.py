import os
import sys
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
import albumentations as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CUSTOM_CLASSES = ["box"]
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

training_augmentation = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.35),
        A.Sharpen(p=0.35),
        A.GaussianBlur(p=0.35),
        A.Resize(height=320, width=320)
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

eval_augmentation = A.Compose(
    [
        A.Resize(height=320, width=320)
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
)

def training_transforms(input, target):
    if len(target) == 0:
        return None, None
    bbox_list = [ann['bbox'] for ann in target]
    category_id_list = [ann['category_id'] for ann in target]
    augmented = training_augmentation(image=np.array(input), bboxes=bbox_list, category_ids=category_id_list)
    input = torch.from_numpy(augmented['image']).permute(2, 0, 1).float() / 255.0
    bbox_list_coco = augmented['bboxes']
    bbox_list_xyxy = []
    if bbox_list_coco:
        for box in bbox_list_coco:
            xmin, ymin, w, h = box
            xmax = xmin + w
            ymax = ymin + h
            bbox_list_xyxy.append([xmin, ymin, xmax, ymax])
    category_id_list = augmented['category_ids']
    new_target = {}
    new_target['labels'] = torch.tensor(category_id_list).to(torch.int64)
    new_target['boxes'] = torch.tensor(bbox_list_xyxy)
    return input, new_target

def eval_transforms(input, target):
    if len(target) == 0:
        return None, None
    bbox_list = [ann['bbox'] for ann in target]
    category_id_list = [ann['category_id'] for ann in target]
    augmented = eval_augmentation(image=np.array(input), bboxes=bbox_list, category_ids=category_id_list)
    input = torch.from_numpy(augmented['image']).permute(2, 0, 1).float() / 255.0
    bbox_list_coco = augmented['bboxes']
    bbox_list_xyxy = []
    if bbox_list_coco:
        for box in bbox_list_coco:
            xmin, ymin, w, h = box
            xmax = xmin + w
            ymax = ymin + h
            bbox_list_xyxy.append([xmin, ymin, xmax, ymax])
    category_id_list = augmented['category_ids']
    new_target = {}
    new_target['labels'] = torch.tensor(category_id_list).to(torch.int64)
    new_target['boxes'] = torch.tensor(bbox_list_xyxy)
    return input, new_target

training_dataset = CocoDetection(
    root='../data/bouling box.v5i.coco/train',
    annFile='../data/bouling box.v5i.coco/train/_annotations.coco.json',
    transforms=training_transforms
)

eval_dataset = CocoDetection(
    root='../data/bouling box.v5i.coco/valid',
    annFile='../data/bouling box.v5i.coco/valid/_annotations.coco.json',
    transforms=eval_transforms
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
        for batch in tqdm(dataloader, file=sys.stdout):
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}\n")
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
        for batch in tqdm(dataloader, file=sys.stdout):
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