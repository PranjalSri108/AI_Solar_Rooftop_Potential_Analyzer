import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch.optim.lr_scheduler import StepLR

sys.path.append('/home/robosobo/solar_rooftop/Datasets')

CSV_PATH = 'Datasets/solar_train.csv'
DATA_DIR = 'Datasets/'

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 15 
IMG_SIZE = 320

ENCODER = 'timm-efficientnet-b3'
WEIGHTS = 'imagenet'

device = torch.device('cpu')

def clean_dataset(df):
    def fix_path(path):
        if path.startswith('images') and '_label' in path:
            return path.replace('images', 'masks', 1)
        return path

    df['images'] = df['images'].apply(fix_path)
    df['masks'] = df['masks'].apply(fix_path)
    
    df = df[df['images'].apply(lambda x: os.path.exists(os.path.join(DATA_DIR, x)))]
    df = df[df['masks'].apply(lambda x: os.path.exists(os.path.join(DATA_DIR, x)))]
    
    return df

print("Loading and cleaning dataset...")
df = pd.read_csv(CSV_PATH)
df = clean_dataset(df)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
print("Dataset loaded and cleaned.")

def get_train_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.5),
    ]) 

def get_valid_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ])

class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]

            image_path = os.path.join(DATA_DIR, row.images)
            mask_path = os.path.join(DATA_DIR, row.masks)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")

            mask = np.expand_dims(mask, axis=-1)

            if self.augmentations:
                augmented = self.augmentations(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

            image = torch.Tensor(image) / 255.0
            mask = torch.round(torch.Tensor(mask) / 255.0)

            return image, mask
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

print("Creating datasets...")
trainset = SegmentationDataset(train_df, augmentations=get_train_augs())
validset = SegmentationDataset(valid_df, augmentations=get_valid_augs())

print(f'Length of trainset: {len(trainset)}')
print(f'Length of validset: {len(validset)}')

print("Creating data loaders...")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

print(f'Total no. of batches in trainloader: {len(trainloader)}')
print(f'Total no. of batches in validloader: {len(validloader)}')

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.backbone = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.backbone(images)

        if masks is not None:
            dice_loss = DiceLoss(mode='binary')(logits, masks)
            bce_loss = nn.BCEWithLogitsLoss()(logits, masks)
            focal_loss = FocalLoss(mode='binary')(logits, masks)
            return logits, dice_loss + bce_loss + 0.5 * focal_loss

        return logits

print("Creating model...")
model = SegmentationModel().to(device)
print("Model created.")

def train_fn(dataloader, model, optimizer):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Train"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_fn(dataloader, model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Valid"):
            images, masks = images.to(device), masks.to(device)
            logits, loss = model(images, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

print("Creating optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
print("Optimizer created.")

print("Creating learning rate scheduler...")
scheduler = StepLR(optimizer, 15, gamma=0.1)
print("Scheduler created.")

best_loss = float('inf')

print("Starting training loop...")
for i in range(EPOCHS):
    print(f"Epoch {i+1}/{EPOCHS}")
    train_loss = train_fn(trainloader, model, optimizer)
    valid_loss = eval_fn(validloader, model)

    if valid_loss < best_loss:
        torch.save(model.state_dict(), 'best-model.pt')
        print("SAVED MODEL")
        best_loss = valid_loss

    print(f'Epoch: {i+1}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, LR: {scheduler.get_last_lr()[0]}')
    
    scheduler.step()

print("Training completed.")

# Visualization
print("Starting visualization...")
idx = 25

model.load_state_dict(torch.load('best-model.pt', map_location=device))
image, mask = validset[idx]
image, mask = image.to(device), mask.to(device)

logits_mask = model(image.unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0

image = image.cpu()
mask = mask.cpu()
pred_mask = pred_mask.cpu()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze(), cmap='gray')
plt.title('Ground Truth Mask')
plt.subplot(1, 3, 3)
plt.imshow(pred_mask.squeeze().detach().numpy(), cmap='gray')
plt.title('Predicted Mask')
plt.show()
print("Visualization completed.")