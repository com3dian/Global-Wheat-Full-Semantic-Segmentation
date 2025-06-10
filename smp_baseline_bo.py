
'''

class_pixel_counts = {0: 8,570,875, 3: 13,707,730, 1: 2,768,275, 2: 905,376}

0 - background
1 - head
2 - stem
3 - leaf

'''

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm


from torch.utils.data import DataLoader
from gwfss_dataset import GWFSSDataset
from segmentation_models_pytorch import Unet
from torchmetrics.segmentation import MeanIoU
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from unbalanced_loss import ClassBalancedCELoss, get_samples_per_class, get_class_balanced_weights

### For Bayesian optimization
from skopt.utils import use_named_args
from skopt.space import Integer, Categorical, Real
from skopt import gp_minimize


class_pixel_counts = {0: 8570875, 3: 13707730, 1: 2768275, 2: 905376}
class_names = ['background', 'head', 'stem', 'leaf']


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


def train_and_evaluate(args, model, train_loader, val_loader, criterion, optimizer, metric, epochs):
    '''
    Trains the model and evaluates on the validation set.
    '''
    model.train()
    for epoch in tqdm(range(epochs), desc="Training", disable=True):
        for i, (images, class_ids, _) in enumerate(train_loader):
            images = images.to(args.device)
            class_ids = class_ids.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, class_ids)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        for i, (images, class_ids, _) in enumerate(tqdm(val_loader, desc="Validating", disable=True)):
            images = images.to(args.device)
            class_ids = class_ids.to(args.device)
            outputs = model(images) # shape (batch_size, num_classes, height, width)
            preds = torch.argmax(outputs, dim=1)
            score = metric(preds.cpu(), class_ids.cpu())
        metric_avg = metric.compute()
        metric.reset() # reset the metric accumulator for the next epoch
    return metric_avg


def cross_validate(args, dataset, k_folds, batch_size, epochs, class_weights, learning_rate):
    '''
    Performs k-fold cross-validation.
    '''

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)  # Set random_state for reproducibility
    cv_scores = []
    eval_metric = MeanIoU(num_classes=dataset.num_classes, per_class=True, input_format='index') # set format type to index when setting per_class to True. Index corresponds to the class id. We dont use one-hot encoding.

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{k_folds}")

        model = Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=dataset.num_classes,
        )
        model.to(args.device)

        ##### Compute class weights #####
        # hardcoded class weights.
        # class_weights = torch.tensor([class_pixel_counts[i] for i in range(dataset.num_classes)], dtype=torch.float32)
        # class_weights = class_weights / class_weights.sum()
        # class_weights = 1/class_weights # we give high weight to the classes with less pixels.
        # class_weights = class_weights.to(args.device)

        # class balanced loss function Cui et al. 
        samples_per_class = get_samples_per_class(dataset)
        weights = get_class_balanced_weights(samples_per_class, args.class_weights)
        print(weights)
        weights = weights.to(args.device)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        metric = MeanIoU(num_classes=dataset.num_classes)

        # Create data loaders for the current fold
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Train and evaluate
        accuracy = train_and_evaluate(args, model, train_loader, val_loader, criterion, optimizer, eval_metric, epochs)

        # print(f"Fold {fold+1} Background Validation Accuracy: {accuracy[0]:.4f}")
        # print(f"Fold {fold+1} Head Validation Accuracy: {accuracy[1]:.4f}")
        # print(f"Fold {fold+1} Stem Validation Accuracy: {accuracy[2]:.4f}")
        # print(f"Fold {fold+1} Leaf Validation Accuracy: {accuracy[3]:.4f}")
        # print(f"Fold {fold+1} Mean Validation Accuracy: {torch.mean(accuracy):.4f}")

        cv_scores.append(torch.mean(accuracy))

    return cv_scores, model

if __name__ == "__main__":

    n_calls = 200  # You can increase this for more thorough optimization
    # --- 2. Define the search space for hyperparameters ---
    space = [
        Integer(8, 32, name='batch_size'),  # Batch size for training
        Integer(20, 50, name='epochs'),  # Number of training epochs
        Real(0.0001, 0.01, name='learning_rate')
    ]

    @use_named_args(space)
    def objective(batch_size, epochs, learning_rate):

        batch_size = int(batch_size)
        epochs = int(epochs)

        print(f"\nSampled values: batch_size = {batch_size}, epochs = {epochs}, learning_rate = {learning_rate}")

        parser = argparse.ArgumentParser()
        parser.add_argument("--visualize_predictions", action="store_true", default=False)
        parser.add_argument("--root_dir", type=str,
                            default="/lustre/scratch/WUR/AIN/nedun001/Global-Wheat-Full-Semantic-Segmentation/data")
        parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
        parser.add_argument("--k_folds", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--class_weights", type=float, default=0.999)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        args = parser.parse_args()

        k_folds = args.k_folds
        args.batch_size = batch_size
        args.epochs = epochs
        args.learning_rate = learning_rate

        train_dataset = GWFSSDataset(root_dir=args.root_dir, split='train')
        # test_dataset = GWFSSDataset(root_dir=args.root_dir, split='val')  # no masks available.

        cv_scores, model = cross_validate(args, train_dataset, k_folds, args.batch_size, args.epochs, args.class_weights, args.learning_rate)
        mean_iou = 1 - np.mean(cv_scores)
        return mean_iou


    print(f"\nRunning Bayesian Optimization for {n_calls} iterations...")
    res_gp = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=10,  # the number of random initialization points
        random_state = 44,
        verbose=True  # Set to False to suppress iteration details
    )

    # --- 5. Print the best hyperparameters found ---
    print("\n--- Bayesian Optimization Results ---")
    print(f"Best validation loss found: {res_gp.fun:.4f}")
    print("Best hyperparameters:")
    print(f"  batch_size: {res_gp.x[0]}")
    print(f"  epochs: {res_gp.x[1]}")
    print(f"  learning rate: {res_gp.x[2]}")

    # # You can also access the best parameters by their names
    # best_batch_size, best_epochs, class_weights,  learning_rate= res_gp.x
    # print(f"Optimal batch_size: {best_batch_size}")
    # print(f"Optimal epochs: {best_epochs}")
    # print(f"Optimal class weights: {class_weights}")
    # print(f"Optimal learning rate: {learning_rate}")

    print("\nBayesian Optimization script finished.")




