import torch
from torch import nn


class UnlikelihoodLoss(nn.Module):
    def __init__(self, ignore_index=-100, epsilon=1e-10) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred_values, target_values):
        loss = 0.0
        n, _ = pred_values.shape
        for pred, target in zip(pred_values, target_values):
            class_index = int(target.item())
            if class_index == self.ignore_index:
                n -= 1
                continue
            prob_pred = torch.exp(pred[class_index]) / (torch.exp(pred).sum())
            loss = loss + torch.log(1 - prob_pred + self.epsilon)
        loss = -loss / n
        return loss

    def __call__(self, pred_values, target_values):
        return self.forward(pred_values, target_values)


class CrossEntropyAndUnlikelihoodLoss(nn.Module):
    def __init__(self, ignore_index=-100, epsilon=1e-10) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred_values, target_values, known_labels, unknown_labels):
        loss = 0.0
        n, _ = pred_values.shape
        for pred, target, known_label, unknown_label in zip(
            pred_values, target_values, known_labels, unknown_labels
        ):
            class_index = int(target.item())
            if class_index == self.ignore_index:
                n -= 1
                continue
            prob_pred = torch.exp(pred[class_index]) / (torch.exp(pred).sum())
            loss = (
                loss
                + torch.log(prob_pred) * known_label
                + torch.log(1 - prob_pred + self.epsilon) * unknown_label
            )
        loss = -loss / n
        return loss

    def __call__(self, pred_values, target_values, known_labels, unknown_labels):
        return self.forward(pred_values, target_values, known_labels, unknown_labels)
