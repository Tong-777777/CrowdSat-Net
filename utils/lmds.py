# Local Maxima Detection Strategy from https://github.com/dk-liang/FIDTM (author: dklinag)

import torch
import numpy
import torch.nn.functional as F
from typing import Tuple, List

class LMDS:
    def __init__(
        self,
        kernel_size: tuple = (3, 3),
        adapt_ts: float = 100.0 / 255.0,
        neg_ts: float = 0.1
        ) -> None:
        '''
        Args:
            kernel_size (tuple, optional): size of the kernel used to select local
                maxima. Defaults to (3,3) (as in the paper).
            adapt_ts (float, optional): adaptive threshold to select final points
                from candidates. Defaults to 100.0/255.0 (as in the paper).
            neg_ts (float, optional): negative sample threshold used to define if
                an image is a negative sample or not. Defaults to 0.1 (as in the paper).
        '''
        assert kernel_size[0] == kernel_size[1], \
            f'The kernel shape must be a square, got {kernel_size[0]}x{kernel_size[1]}'
        assert not kernel_size[0] % 2 == 0, \
            f'The kernel size must be odd, got {kernel_size[0]}'

        self.kernel_size = tuple(kernel_size)
        self.adapt_ts = adapt_ts
        self.neg_ts = neg_ts


    def __call__(self, est_map: torch.Tensor) -> Tuple[list, list, list, list]:
        '''
        Args:
            est_map (torch.Tensor): the estimated FIDT map

        Returns:
            Tuple[list,list,list,list]
                counts, labels, scores and locations per batch
        '''

        batch_size, classes = est_map.shape[:2]

        b_counts, b_labels, b_scores, b_locs = [], [], [], []
        for b in range(batch_size):
            counts, labels, scores, locs = [], [], [], []

            for c in range(classes):
                count, loc, score = self._lmds(est_map[b][c])
                counts.append(count)
                labels = [*labels, *[c + 1] * count]
                scores = [*scores, *score]
                locs = [*locs, *loc]

            b_counts.append(counts)
            b_labels.append(labels)
            b_scores.append(scores)
            b_locs.append(locs)

        return b_counts, b_locs, b_labels, b_scores


    def _local_max(self, est_map: torch.Tensor) -> torch.Tensor:
        ''' Shape: est_map = [B,C,H,W] '''

        pad = int(self.kernel_size[0] / 2)
        keep = torch.nn.functional.max_pool2d(est_map, kernel_size=self.kernel_size, stride=1, padding=pad)
        keep = (keep == est_map).float()
        est_map = keep * est_map

        return est_map

    def _get_locs_and_scores(
            self,
            locs_map: torch.Tensor,
            scores_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Shapes: locs_map = [H,W] and scores_map = [H,W] '''

        locs_map = locs_map.data.cpu().numpy()
        scores_map = scores_map.data.cpu().numpy()
        locs = []
        scores = []
        for i, j in numpy.argwhere(locs_map == 1):
            locs.append((i, j))
            scores.append(scores_map[i][j])

        return torch.Tensor(locs), torch.Tensor(scores)


    def _lmds(self, est_map: torch.Tensor) -> Tuple[int, list, list]:
        ''' Shape: est_map = [H,W] '''
        est_map_max = torch.max(est_map).item()

        # local maxima
        est_map = self._local_max(est_map.unsqueeze(0).unsqueeze(0))

        # adaptive threshold for counting
        est_map[est_map < self.adapt_ts * est_map_max] = 0
        scores_map = torch.clone(est_map)
        est_map[est_map > 0] = 1

        # negative sample
        if est_map_max < self.neg_ts:
            est_map = est_map * 0

        # count
        count = int(torch.sum(est_map).item())

        # locations and scores
        locs, scores = self._get_locs_and_scores(
            est_map.squeeze(0).squeeze(0),
            scores_map.squeeze(0).squeeze(0)
        )

        return count, locs.tolist(), scores.tolist()
