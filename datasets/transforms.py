import PIL
from PIL import Image
import numpy
import torch
import torchvision
import scipy

from typing import Dict, Optional, Union, Tuple, List, Any

from utils.registry import Registry

TRANSFORMS = Registry(name = 'transforms', module_key = 'datasets.transforms')

__all__ = ['TRANSFORMS', *TRANSFORMS.registry_names]

def _point_buffer(x: int, y: int, mask: torch.Tensor, radius: int) -> torch.Tensor:
    x_t, y_t = torch.arange(0, mask.size(1)), torch.arange(0, mask.size(0))
    buffer = (x_t.unsqueeze(0) - x) ** 2 + (y_t.unsqueeze(1) - y) ** 2 < radius ** 2
    return buffer

@TRANSFORMS.register()
class MultiTransformsWrapper:
    ''' Independently applies each input transformation to the called input and
    returns the results separately in the same order as the specified transforms

    Args:
        transforms(list): list of transforms that take image (PIL or Tensor) and
            target (dict) as inputs
    '''

    def __init__(self, transforms: List[object]) -> None:
        self.transforms = transforms

    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        '''
        Args:
            image (PIL.Image.Image or torch.Tensor): image of reference [C,H,W], only for
                pipeline convenience
            target (dict): corresponding target containing at least 'points' and 'labels'
                keys, with torch.Tensor as value. Labels must be integers!

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor]]:
                the transormed image and the tuple of transformed outputs in the same
                order as the specified transforms
        '''

        outputs = []
        for trans in self.transforms:
            img, tr_trgt = trans(image, target)
            outputs.append(tr_trgt)

        return img, tuple(outputs)

@TRANSFORMS.register()
class SampleToTensor:
    "Convert image and target to Tensors"

    def __call__(
            self,
            img: Union[Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor],
            crow_type: str = 'point'
            ) -> Tuple[torch.Tensor, str]:
        """
        :param img: PIL image with [C, H, W] shape
        :param target: corresponding target
        :param crowd_type: crowd-labeled type, including point, bbox, and density. In this project, point type is used.
        :return: Tuple[torch.Tensor, dict]: the transormed image and target
        """

        tr_img = torchvision.transforms.ToTensor()(img)

        tr_target = {}
        tr_target.update(dict(**target))

        tr_target['points'] = torch.as_tensor(tr_target['points'], dtype=torch.int64)

        tr_target['labels'] = torch.as_tensor(tr_target['labels'], dtype=torch.int64)

        return tr_img, tr_target

@TRANSFORMS.register()
class UnNormalize:
    "Reverse normalization"

    def __init__(
            self,
            mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
            std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
            ) -> None:

        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        for i, m, s in zip(img, self.mean, self.std):
            i.mul_(s).add_(m)

        return img

@TRANSFORMS.register()
class Normalize:
    "normalization"

    def __init__(
            self,
            mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
            std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
            ) -> None:

        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        for i, m, s in zip(img, self.mean, self.std):
            i.sub_(m).div_(s)

        return img

@TRANSFORMS.register()
class DownSample:
    "DownSample img by a ratio "

    def __init__(
            self,
            down_ratio: int = 2,
            crowd_type: str = 'point'
            ) -> None:

        assert crowd_type in ['bbox', 'point'], \
            f'Annotations type must be \'bbox\' or \'point\', got \'{crowd_type}\''

        self.down_ratio = down_ratio
        self.crowd_type = crowd_type

    def __call__(
            self,
            img: Union[Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:

        if isinstance(img, PIL.Image.Image):
            img = torchvision.transforms.ToTensor()(img)

        target['points'] = torch.div(target['points'], self.down_ratio, rounding_mode='floor')

        return img, target

@TRANSFORMS.register()
class PointsToMask:
    "Convert points annotation to mask with a buffer option"
    "based on https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/data/transforms.py"

    def __init__(
            self,
            radius: int = 1,
            num_classes: int = 2,
            onehot: bool = False,
            squeeze: bool = True,
            down_ratio: Optional[int] = None,
            target_type: str = 'long'
    ) -> None:
        '''
        Args:
            radius (int, optional): buffer (pixel radius) to define a point in
                the mask. Defautls to 1 (i.e. non buffer)
            num_classes (int, optional): number of classes, background included.
                Defaults to 2
            onehot (bool, optional): set to True do enable one-hot encoding.
                Defaults to False
            squeeze (bool, optional): when onehot is False, set to True to squeeze the
                mask to get a Tensor of shape [H,W], otherwise the returned mask has
                a shape of [1,H,W].
                Defaults to False
            down_ratio (int, optional): if specified, the target will be downsampled
                according to the ratio.
                Defaults to None
            target_type (str, optional): output data type of target. Defaults to 'long'.
        '''

        assert target_type in ['long', 'float'], \
            f"target type must be either 'long' or 'float', got {target_type}"

        self.radius = radius
        self.num_classes = num_classes - 1
        self.onehot = onehot
        self.squeeze = squeeze
        self.down_ratio = down_ratio
        self.target_type = target_type


    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            image (PIL.Image.Image or torch.Tensor): image of reference [C,H,W], only for
                pipeline convenience
            target (dict): corresponding target containing at least 'points' and 'labels'
                keys, with torch.Tensor as value. Labels must be integers!

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                the transormed image and the mask
        '''

        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        self.img_height, self.img_width = image.size(1), image.size(2)
        if self.down_ratio is not None:
            self.img_height = self.img_height // self.down_ratio
            self.img_width = self.img_width // self.down_ratio
            _, target = DownSample(down_ratio=self.down_ratio, crowd_type='point')(
                image, target.copy()
            )

        mask = torch.zeros((1, self.img_height, self.img_width)).long()

        # fill the mask
        if len(target['points']) > 0:
            for point, label in zip(target['points'], target['labels']):
                x, y = point[0], point[1]
                point_buffer = _point_buffer(x, y, mask[0], self.radius)
                mask[0, point_buffer] = label

        if self.onehot:
            mask = self._onehot(mask)

        if self.squeeze:
            mask = mask.squeeze(0)

        if self.target_type == 'float':
            mask = mask.float()

        return image, mask

    def _onehot(self, mask: torch.Tensor):
        onehot_mask = torch.nn.functional.one_hot(mask, self.num_classes + 1)
        onehot_mask = torch.movedim(onehot_mask, -1, -3)
        return onehot_mask


@TRANSFORMS.register()
class FIDT:
    ''' Convert points annotations into Focal-Inverse-Distance-Transform map.

    In case of multi-class, returns one-hot encoding masks.

    For binary case, you can let the num_classes argument by default, this will return a
    density map of one channel only [1, H, W].

    Inspired from:
    Liang et al. (2021) - "Focal Inverse Distance Transform Maps for Crowd Localization
    and Counting in Dense Crowd"
    '''

    def __init__(
            self,
            alpha: float = 0.02,
            beta: float = 0.75,
            c: float = 1.0,
            radius: int = 1,
            num_classes: int = 2,
            add_bg: bool = False,
            down_ratio: Optional[int] = None
    ) -> None:
        '''
        Args:
            alpha (float, optional): parameter, can be adjusted. Defaults to 0.02
            beta (float, optional): parameter, can be adjusted. Defaults to 0.75
            c (float, optional): parameter, can be adjusted. Defaults to 1.0
            radius (int, optional): buffer (pixel radius) to define a point in
                the mask. Defautls to 1 (i.e. non buffer)
            num_classes (int, optional): number of classes, background included. If
                higher than 2, returns one-hot encoding masks [C, H, W], otherwise
                returns a binary mask [1, H, W] even if different categories of labels
                are called. Defaults to 2
            add_bg (bool, optional): set to True to add background map in any case. It
                is built by substracting all positive locations from ones tensor.
                Defaults to False
            down_ratio (int, optional): if specified, the target will be downsampled
                according to the ratio.
                Defaults to None
        '''

        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.radius = radius
        self.num_classes = num_classes - 1
        self.add_bg = add_bg
        self.down_ratio = down_ratio

    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        target: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            image (PIL.Image.Image or torch.Tensor): image of reference [C,H,W], only for
                pipeline convenience
            target (dict): corresponding target containing at least 'points' and 'labels'
                keys, with torch.Tensor as value. Labels must be integers!

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                the transormed image and the FIDT map(s)
        '''

        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        self.img_height, self.img_width = image.size(1), image.size(2)
        if self.down_ratio is not None:
            self.img_height = self.img_height // self.down_ratio
            self.img_width = self.img_width // self.down_ratio
            _, target = DownSample(down_ratio=self.down_ratio, crowd_type='point')(
                image, target.copy()
            )

        if self.num_classes == 1:
            new_target = target.copy()
            new_target.update(labels=[1] * len(new_target['labels']))
            dist_map = self._onehot(image, new_target)
        else:
            dist_map = self._onehot(image, target)

        if self.add_bg:
            dist_map = self._add_background(dist_map)

        return image, dist_map.type(image.type())

    def _get_fidt(self, mask: torch.Tensor) -> torch.Tensor:

        dist_map = scipy.ndimage.distance_transform_edt(mask)
        dist_map = torch.from_numpy(dist_map)
        dist_map = 1 / (torch.pow(dist_map, self.alpha * dist_map + self.beta) + self.c)
        dist_map = torch.where(dist_map < 0.01, 0., dist_map)

        return dist_map

    def _onehot(self, image: torch.Tensor, target: torch.Tensor):

        dist_maps = torch.zeros((self.num_classes, self.img_height, self.img_width))

        if len(target['points']) > 0:
            labels = numpy.unique(target['labels'])
            masks = torch.ones((self.num_classes, self.img_height, self.img_width))

            for point, label in zip(target['points'], target['labels']):
                x, y = point[0], point[1]
                point_buffer = _point_buffer(x, y, masks[label - 1], self.radius)
                masks[label - 1, point_buffer] = 0

            dist_maps = torch.ones((self.num_classes, self.img_height, self.img_width), dtype=torch.float64)
            for i, mask in enumerate(masks):
                mask = self._get_fidt(mask)
                if i + 1 in labels:
                    dist_maps[i] = mask
                else:
                    dist_maps[i] = torch.zeros((self.img_height, self.img_width), dtype=torch.float64)

        return dist_maps

    def _add_background(self, dist_map: torch.Tensor) -> torch.Tensor:
        background = torch.ones((1, *dist_map.shape[1:]))
        merged_dist = dist_map.sum(dim=0, keepdim=True)
        background = torch.sub(background, merged_dist)
        output = torch.cat((background, dist_map), dim=0)
        return output
