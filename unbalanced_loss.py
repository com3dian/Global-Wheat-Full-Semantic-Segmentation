import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_class_balanced_weights(samples_per_class, beta):
    """
    \frac{1 - \beta}{1 - \beta^{N_y}}
    where N_y is the number of samples for class y.
    Then normalize the weights so that they sum to 1.
    return shape: (num_classes,)
    """

    # if samples_per_class is too large, then beta ** samples_per_class will tend to 0. 
    # hence we scale down the samples_per_class. we can divide by the sum
    samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)
    samples_per_class = samples_per_class / torch.sum(samples_per_class)
    
    effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(samples_per_class, dtype=torch.float32))
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights) * len(samples_per_class)  # Normalize
    return weights

def get_samples_per_class(dataset):
    sample_per_each_class = [0, 0, 0, 0]

    for data in dataset:
        _, class_id_img, _ = data
        class_labels, counts = np.unique(class_id_img, return_counts=True)
        for label, count in zip(class_labels, counts):
            sample_per_each_class[label] += count

    return sample_per_each_class


class ClassBalancedCELoss(nn.Module):
    """
    # criterion = nn.CrossEntropyLoss()
    samples_per_class = get_samples_per_class(train_dataset)
    beta = 0.9999
    # For cross-entropy:
    criterion = ClassBalancedCELoss(samples_per_class, beta)
    """
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        self.weights = get_class_balanced_weights(samples_per_class, beta)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        self.weights = get_class_balanced_weights(samples_per_class, beta)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        cb_weights = self.weights.to(logits.device)[targets]
        focal_weights = (1 - probs.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma

        loss = -cb_weights * focal_weights * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return loss.mean()
    
if __name__ == "__main__":
    samples_per_class = [8570875, 2768275, 905376, 13707730]
    beta = 0.999
    weights = get_class_balanced_weights(samples_per_class, beta)
    print(weights)