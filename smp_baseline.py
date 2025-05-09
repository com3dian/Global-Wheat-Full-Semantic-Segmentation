
'''
TODO:


Current code:

- create a cv fold (stratified split based on pixel count).
- iou  per class (parameter to mean iou metric)
- weighted loss function (torch and class balanced paper)
- output the confusion matrix


SSL part :
- add domain information to the model as an auxiliary head.



'''

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


def visualize_predictions(model, val_loader, val_dataset):
    model.eval()
    with torch.no_grad():
        for idx, (images, _, _) in enumerate(val_loader):
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).squeeze(0)

            #store the rgb colors for each class
            save_img = np.zeros((preds.shape[0], preds.shape[1], 3))
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    color = val_dataset.class_info['colors'][preds[i, j]][::-1]
                    save_img[i, j, :] = color
            
            save_img = save_img / 255.0
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
            plt.subplot(1, 2, 2)
            plt.imshow(save_img)
            plt.savefig(f"/Users/vishalned/Desktop/GWFSS/gwfss_competition_val/preds/{idx}.png")
            plt.close()




def main(args):
    train_dataset = GWFSSDataset(root_dir=args.root_dir, split='train')
    test_dataset = GWFSSDataset(root_dir=args.root_dir, split='val') # no masks available.

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=train_dataset.num_classes,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metric = MeanIoU(num_classes=train_dataset.num_classes)

    ##### Visualize predictions of the model #####
    if args.visualize_predictions:
        print("Visualizing predictions of the model")
        model.load_state_dict(torch.load("model.pth"))
        visualize_predictions(model, test_loader, test_dataset)
        return
    

    ##### Training loop #####
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (images, class_ids, _) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, class_ids)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)   
            metric.update(preds, class_ids)
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        metric_avg = metric.compute()
        print(f"Average Mean IoU: {metric_avg:.4f}")
        metric.reset() # reset the metric accumulator for the next epoch

    # save the model
    torch.save(model.state_dict(), "model.pth")
    ##### End of training loop #####



   

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_predictions", action="store_true", default=False)
    parser.add_argument("--root_dir", type=str, default="/Users/vishalned/Desktop/GWFSS/")
    args = parser.parse_args()


    main(args)