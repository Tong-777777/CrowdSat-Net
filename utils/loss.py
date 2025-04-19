from utils.registry import Registry
import torch
from typing import Optional

LOSSES = Registry('losses')

@LOSSES.register()
class FocalLoss(torch.nn.Module):

    def __init__(
            self,
            alpha: int = 2,
            beta: int = 4,
            reduction: str = 'sum',
            weights: Optional[torch.Tensor] = None,
            density_weight: Optional[str] = None,
            normalize: bool = False,
            eps: float = 1e-6
            ) -> None:

        super().__init__()

        assert reduction in ['mean', 'sum'], \
            f'Reduction must be either \'mean\' or \'sum\', got {reduction}'

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.weights = weights
        self.density_weight = density_weight
        self.normalize = normalize
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param output:
        :param target:
        :return: torch.Tensor
        """
        B, C, _, _ = target.shape

        if self.weights is not None:
            assert self.weights.shape[0] == C, \
                'Number of weights must match the number of channels, ' \
                f'got {C} channels and {self.weights.shape[0]} weights'

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = torch.zeros((B, C))

        # avoid NaN when net output is 1.0 or 0.0
        output = torch.clamp(output, min=self.eps, max=1 - self.eps)

        pos_loss = torch.log(output) * torch.pow(1 - output, self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum(3).sum(2)
        pos_loss = pos_loss.sum(3).sum(2)
        neg_loss = neg_loss.sum(3).sum(2)

        for b in range(B):
            for c in range(C):
                density = torch.tensor([1]).to(neg_loss.device)
                if self.density_weight == 'linear':
                    density = num_pos[b][c]
                elif self.density_weight == 'squared':
                    density = num_pos[b][c] ** 2
                elif self.density_weight == 'cubic':
                    density = num_pos[b][c] ** 3

                if num_pos[b][c] == 0:
                    loss[b][c] = loss[b][c] - neg_loss[b][c]
                else:
                    loss[b][c] = density * (loss[b][c] - (pos_loss[b][c] + neg_loss[b][c]))
                    if self.normalize:
                        loss[b][c] = loss[b][c] / num_pos[b][c]

        if self.weights is not None:
            loss = self.weights * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()






