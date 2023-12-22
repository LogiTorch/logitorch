import torch
from torch import nn


class UnlikelihoodLoss(nn.Module):
    """
    Calculates the unlikelihood loss for a given prediction and target values.

    Args:
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to -100.
        epsilon (float, optional): Small value added to the denominator to avoid division by zero. Defaults to 1e-10.
    """

    def __init__(self, ignore_index=-100, epsilon=1e-10) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred_values, target_values):
        """
        Forward pass of the unlikelihood loss calculation.

        Args:
            pred_values (torch.Tensor): Predicted values.
            target_values (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Calculated loss.
        """
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
    """
    Calculates the cross-entropy and unlikelihood loss for a given prediction and target values.

    Args:
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to -100.
        epsilon (float, optional): Small value added to the denominator to avoid division by zero. Defaults to 1e-10.
    """

    def __init__(self, ignore_index=-100, epsilon=1e-10) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred_values, target_values, known_labels, unknown_labels):
        """
        Forward pass of the cross-entropy and unlikelihood loss calculation.

        Args:
            pred_values (torch.Tensor): Predicted values.
            target_values (torch.Tensor): Target values.
            known_labels (torch.Tensor): Known labels.
            unknown_labels (torch.Tensor): Unknown labels.

        Returns:
            torch.Tensor: Calculated loss.
        """
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
