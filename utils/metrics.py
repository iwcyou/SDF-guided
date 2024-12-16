import torch
import torch.nn as nn
import numpy as np


class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def forward(self, target, input):
        eps = 1e-10

        # Apply thresholding
        input_ = (input > self.threshold).float()  # Convert to binary: 1 if greater than threshold, else 0
        target_ = (target > self.threshold).float()

        # Calculate intersection and union
        intersection = input_ * target_  # Element-wise multiplication for intersection
        union = input_ + target_  # Element-wise addition for union
        union = torch.clamp(union, 0, 1)  # Ensure union doesn't exceed 1

        # Calculate metrics per image in the batch
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))  # Sum over batch, height, and width
        union_sum = torch.sum(union, dim=(1, 2, 3))
        target_sum = torch.sum(target_, dim=(1, 2, 3))  # Sum over target for recall calculation
        input_sum = torch.sum(input_, dim=(1, 2, 3))  # Sum over input for precision calculation

        # Avoid division by zero by adding eps
        iou = intersection_sum / (union_sum + eps)
        recall = intersection_sum / (target_sum + eps)
        precision = intersection_sum / (input_sum + eps)

        # Calculate accuracy (pixel-level equality)
        correct_pixels = torch.sum(input_ == target_, dim=(1, 2, 3)).float()
        total_pixels = torch.numel(input_) / input_.size(0)  # Total number of pixels per image
        acc = correct_pixels / total_pixels  # Pixel-level accuracy

        # Calculate F1 score
        f1 = 2 * recall * precision / (recall + precision + eps)

        # Average the results over the batch
        acc_mean = torch.mean(acc)
        iou_mean = torch.mean(iou)
        recall_mean = torch.mean(recall)
        precision_mean = torch.mean(precision)
        f1_mean = torch.mean(f1)

        # Return the results as a tensor
        return torch.Tensor([acc_mean, recall_mean, precision_mean, iou_mean, f1_mean])

