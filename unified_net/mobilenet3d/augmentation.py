import torch
import torch.nn as nn
from einops import rearrange
from kornia.augmentation import *
from typing import Union, Tuple
from hyperbox.mutables.spaces import OperationSpace
from hyperbox.networks.base_nas_network import BaseNASNetwork

__all__ = [
    'Base2dTo3d',
    'RandomInvert3d',
    'RandomGaussianNoise3d',
    'RandomBoxBlur3d',
    'RandomErasing3d',
    'RandomSharpness3d',
    'RandomResizedCrop3d',
    'DataAugmentation',
    'DAOperation3D',
]


class Base2dTo3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs = x.shape[0]
        assert len(x.shape) == 5, f"len of x.shape should be 5, i.e., B,C,D,H,W"
        x = rearrange(x, 'b c d h w -> (b d) c h w', b=bs)
        x = self.aug(x)
        x = rearrange(x, '(b d) c h w -> b c d h w', b=bs)
        return x


class RandomResizedCrop3d(Base2dTo3d):
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.8, 1.0),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (3.0 / 4.0, 4.0 / 3.0),
        resample: Union[str, int] = 'bilinear',
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ):
        super(RandomResizedCrop3d, self).__init__()
        self.aug = RandomResizedCrop(
            size, scale, ratio, resample, return_transform,
            same_on_batch, align_corners, p, keepdim, cropping_mode
        )


class RandomInvert3d(Base2dTo3d):
    def __init__(
        self,    
        max_val: Union[float, torch.Tensor] = torch.tensor(1.0),
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False
    ):
        '''
        Args:
            max_val: The expected maximum value in the input tensor. The shape has to
            according to the input tensor shape, or at least has to work with broadcasting.
            return_transform: if ``True`` return the matrix describing the transformation applied to each
                input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
            same_on_batch: apply the same transformation across the batch.
            p: probability of applying the transformation.
            keepdim: whether to keep the output shape the same as input (True) or broadcast it
                    to the batch form (False).
        '''
        super(RandomInvert3d, self).__init__()
        self.aug = RandomInvert(max_val, return_transform, same_on_batch, p, keepdim)


class RandomGaussianNoise3d(Base2dTo3d):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False
    ) -> None:
        super(RandomGaussianNoise3d, self).__init__()
        self.aug = RandomGaussianNoise(mean, std, return_transform, same_on_batch, p, keepdim)


class RandomBoxBlur3d(Base2dTo3d):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        border_type: str = "reflect",
        normalized: bool = True,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False
    ):
        super(RandomBoxBlur3d, self).__init__()
        self.aug = RandomBoxBlur(
        kernel_size, border_type, normalized, return_transform, same_on_batch, p, keepdim)


class RandomErasing3d(Base2dTo3d):
    def __init__(
        self,
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ):
        '''
        Args:
            p: probability that the random erasing operation will be performed.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
            same_on_batch: apply the same transformation across the batch.
            keepdim: whether to keep the output shape the same as input (True) or broadcast it
                            to the batch form (False).
        '''
        super(RandomErasing3d, self).__init__()
        self.aug = RandomErasing(
            scale, ratio, value, return_transform, same_on_batch, p, keepdim)


class RandomSharpness3d(Base2dTo3d):
    def __init__(
        self,
        sharpness: Union[torch.Tensor, float, Tuple[float, float], torch.Tensor] = 0.5,
        same_on_batch: bool = False,
        return_transform: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ):
        super(RandomSharpness3d, self).__init__()
        self.aug = RandomSharpness(
            sharpness, same_on_batch, return_transform, p,  keepdim)




def prob_list_gen(func, num_probs=4, probs: list=None, *args, **kwargs):
    if probs is not None:
        return [func(p=p, *args, **kwargs) for p in probs]
    else:
        return [func(p=p, *args, **kwargs) for p in [i*0.25 for i in range(num_probs)]]

def DAOperation3D(
    affine_degree=30, affine_scale=(1.1, 1.5), affine_shears=20,
    rotate_degree=30,
    crop_size=(16,128,128)
):
    ops = {}
    ops['dflip'] = prob_list_gen(RandomDepthicalFlip3D, probs=[0, 0.5, 1], same_on_batch=False)
    ops['hflip'] = prob_list_gen(RandomHorizontalFlip3D, probs=[0, 0.5, 1], same_on_batch=False)
    ops['vflip'] = prob_list_gen(RandomVerticalFlip3D, probs=[0, 0.5, 1], same_on_batch=False)
    # ops['equal'] = prob_list_gen(RandomEqualize3D, probs=[0, 0.5, 1], same_on_batch=False)

    # affine
    ops['affine'] = [nn.Identity()]
    if isinstance(affine_degree, (float, int)):
        # rotation degree
        affine_degree = [affine_degree]
    if isinstance(affine_shears, (float, int)):
        affine_shears = [affine_shears]
    if isinstance(affine_scale[0], (float, int)):
        # scale, similar to zoom in/out
        affine_scale = [affine_scale]
    for ad_ in affine_degree:
        for ash_ in affine_shears:
            for asc_ in affine_scale:
                affine = prob_list_gen(RandomAffine3D, probs=[0.5, 1], same_on_batch=False, degrees=ad_, scale=asc_, shears=ash_) 
                ops['affine'] += affine

    # random crop
    ops['rcrop'] = []
    if isinstance(crop_size, (float, int)):
        # e.g., crop_size = 32
        crop_size = [(crop_size,)*3]
        rcrop = prob_list_gen(RandomCrop3D, same_on_batch=False, size=crop_size)
    elif isinstance(crop_size[0], (float, int)):
        # e.g., crop_size = (16,64,64)
        crop_size = [crop_size]
    for size in crop_size:
        rcrop = [RandomCrop3D(same_on_batch=False, size=size, p=1)]
        ops['rcrop'] += rcrop

    resize_crop = [nn.Identity()]
    for size in crop_size[:1]:
        size = size[1:]
        for scale in [(0.8, 1), (1, 1)]:
            for ratio in [(1, 1), (3/4, 4/3)]:
                resize_crop += prob_list_gen(RandomResizedCrop3d, probs=[0.5, 1], size=size, scale=scale, ratio=ratio)
    ops['resize_crop'] = resize_crop

    boxblur = [nn.Identity()]
    for ks in [(3,3), (5,5)]:
        boxblur += prob_list_gen(RandomBoxBlur3d, probs=[0.5, 1], kernel_size=ks)
    ops['boxnlur'] = boxblur

    invert = [nn.Identity()]
    for val in [0.25, 0.5, 0.75, 1]:
        invert += prob_list_gen(RandomInvert3d, probs=[0.5, 1], max_val=val)
    ops['invert'] = invert

    gauNoise = [nn.Identity()]
    gauNoise += prob_list_gen(RandomGaussianNoise3d, probs=[0, 0.5, 1])
    ops['gauNoise'] = gauNoise

    erase = [nn.Identity()]
    for scale in [(0.02, 0.1), (0.1, 0.33)]:
        for ratio in [(0.3, 3.3)]:
            erase += prob_list_gen(RandomErasing3d, probs=[0.5, 1], scale=scale, ratio=ratio)
    ops['erase'] = erase

    return ops


class DataAugmentation(BaseNASNetwork):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(
        self,
        rotate_degree=30, crop_size=[(32,128,128), (16,128,128)],
        affine_degree=0, affine_scale=(1.1, 1.5), affine_shears=20,
        mean=0.5, std=0.5,
        mask=None
    ):
        super().__init__(mask)
        self.ops = DAOperation3D(affine_degree, affine_scale, affine_shears, rotate_degree, crop_size)
        transforms = []
        for key, value in self.ops.items():
            transforms.append(OperationSpace(candidates=value, key=key, mask=self.mask, reduction='mean'))
        self.transforms = nn.Sequential(*transforms)
        self.mean = mean
        self.std = std

    # @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor, aug=True) -> torch.Tensor:
        if aug:
            for idx, trans in enumerate(self.transforms):
                x = trans(x)  # BxCXDxHxW
        # normalize
        # Todo: compare with no normalization
        x = (x-self.mean)/self.std
        return x

    @property
    def arch(self):
        _arch = []
        for op in self.transforms:
            mask = op.mask
            if 'bool' in str(mask.dtype):
                index = mask.int().argmax()
            else:
                index = mask.float().argmax()
            _arch.append(f"{op.candidates[index]}")
        _arch = '\n'.join(_arch)
        return _arch

