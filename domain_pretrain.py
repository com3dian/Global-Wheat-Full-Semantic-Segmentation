
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gwfss_dataset import GWFSSDataset
from segmentation_models_pytorch import Unet
from torchmetrics.segmentation import MeanIoU
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torchvision
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchvision.models import ResNet18_Weights

def predict_domain(args):
    '''Use a convolutional neural network to predict the domain of the image. 
    In other words, this is a classification task.'''
    train_dataset = GWFSSDataset(root_dir=args.root_dir, split='train')
    test_dataset = GWFSSDataset(root_dir=args.root_dir, split='val') # no masks available.

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    unique_domains = set(train_dataset.domain_info).union(set(test_dataset.domain_info))
    domain_to_idx = {domain: idx for idx, domain in enumerate(unique_domains)}
    idx_to_domain = {idx: domain for domain, idx in domain_to_idx.items()}
    num_domains = len(unique_domains)
    print(f"Number of domains: {num_domains}")

    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_domains)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ## Metrics: Use F1 score, precision and recall per class, and also averaged over all classes
    metric = {
        "precision_macro": MulticlassPrecision(num_classes=num_domains, average="macro"),
        "recall_macro": MulticlassRecall(num_classes=num_domains, average="macro"),
        "f1_macro": MulticlassF1Score(num_classes=num_domains, average="macro"),
        "precision_per_class": MulticlassPrecision(num_classes=num_domains, average=None),
        "recall_per_class": MulticlassRecall(num_classes=num_domains, average=None),
        "f1_per_class": MulticlassF1Score(num_classes=num_domains, average=None),
    }

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (images, _, domain) in enumerate(train_loader):
            domain_ids = [domain_to_idx[d] for d in domain]
            domain_ids = torch.tensor(domain_ids)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, domain_ids)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)   
            for m in metric.values():
                m.update(preds, domain_ids)
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"\n=== Metrics at Epoch {epoch+1} ===")
        for name, m in metric.items():
            score = m.compute()
            if score.ndim == 1:  # per-class metrics
                for cls_id, s in enumerate(score):
                    domain_name = idx_to_domain[cls_id]
                    print(f"{name} [class {domain_name}]: {s:.4f}")
            else:  # macro metrics
                print(f"{name}: {score:.4f}")
            m.reset()

    # save the model
    torch.save(model.state_dict(), "model.pth")
    ##### End of training loop #####



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/Users/tplas/data/GWFSS-competition/")
    args = parser.parse_args()
    predict_domain(args)