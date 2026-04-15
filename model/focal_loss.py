import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    :param gamma: Focusing exponent. 0 reduces to standard cross-entropy. Defaults to 2.0.
    :param class_weights: Optional 1-D tensor of per-class weights (alpha).
    :param label_smoothing: Label smoothing factor in [0.0, 1.0).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self._gamma = gamma
        self._label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("_class_weights", class_weights)
        else:
            self._class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        :param logits: Raw model output of shape (batch_size, num_classes).
        :param targets: Ground-truth class indices of shape (batch_size,).
        :return: Scalar loss tensor.
        """
        num_classes = logits.size(1)

        # Apply label smoothing to the one-hot targets
        with torch.no_grad():
            smoothed_targets = torch.full_like(logits, self._label_smoothing / num_classes)
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self._label_smoothing + self._label_smoothing / num_classes)

        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()

        # p_t: the predicted probability for the true class
        true_class_probabilities = probabilities.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weights = (1.0 - true_class_probabilities) ** self._gamma

        # Cross-entropy with smoothed targets
        per_sample_cross_entropy = -(smoothed_targets * log_probabilities).sum(dim=1)

        # Apply focal weighting
        per_sample_focal_loss = focal_weights * per_sample_cross_entropy

        # Apply per-class alpha weights if provided
        if self._class_weights is not None:
            alpha_weights = self._class_weights[targets]
            per_sample_focal_loss = alpha_weights * per_sample_focal_loss

        return per_sample_focal_loss.mean()
