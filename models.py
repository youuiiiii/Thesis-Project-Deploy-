import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2

CLASS_NAMES = ['ai', 'human']
IMAGE_SIZE = 224

DCT_SIZE = 32
DCT_KEEP_TOPK = 256
DCT_DROPOUT = 0.3
CNN_DROPOUT = 0.3
FUSION_DROPOUT = 0.3


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def extract_dct_features_from_pil(image, dct_size=DCT_SIZE, keep_topk=DCT_KEEP_TOPK):
    gray = image.convert('L')
    gray = gray.resize((dct_size, dct_size))
    gray_np = np.array(gray, dtype=np.float32) / 255.0

    dct = cv2.dct(gray_np)
    dct = np.abs(dct)

    flat = dct.flatten()
    flat_sorted = np.sort(flat)[::-1]
    features = flat_sorted[:keep_topk]

    if len(features) < keep_topk:
        pad_width = keep_topk - len(features)
        features = np.pad(features, (0, pad_width), mode='constant')

    features = np.log1p(features)
    features = features.astype(np.float32)

    return torch.tensor(features, dtype=torch.float32)


def build_cnn_model(num_classes=2):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


class ResNet18WithDCT(nn.Module):
    def __init__(self, num_classes=2, dct_input_dim=256):
        super().__init__()

        backbone = models.resnet18(weights=None)
        cnn_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.cnn_head = nn.Sequential(
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(cnn_feature_dim, 256),
            nn.ReLU(inplace=True)
        )

        self.dct_head = nn.Sequential(
            nn.Linear(dct_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DCT_DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(128, num_classes)
        )

    def forward(self, image_tensor, dct_tensor):
        cnn_features = self.backbone(image_tensor)
        cnn_features = self.cnn_head(cnn_features)

        dct_features = self.dct_head(dct_tensor)

        fused_features = torch.cat([cnn_features, dct_features], dim=1)
        logits = self.classifier(fused_features)

        return logits


def predict_with_cnn(image_pil, model, device):
    transform = get_eval_transform()
    image = image_pil.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()

    return {
        "predicted_label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "prob_ai": float(probs[0].item()),
        "prob_human": float(probs[1].item()),
    }


def predict_with_cnn_dct(image_pil, model, device):
    transform = get_eval_transform()
    image = image_pil.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)
    dct_tensor = extract_dct_features_from_pil(
        image,
        dct_size=DCT_SIZE,
        keep_topk=DCT_KEEP_TOPK
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, dct_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()

    return {
        "predicted_label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "prob_ai": float(probs[0].item()),
        "prob_human": float(probs[1].item()),
    }