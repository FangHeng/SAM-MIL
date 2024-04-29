import numpy as np
import torch
from typing import Optional, List
from torch import Tensor

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

def max_overlap_crop(image, crop_size):
    h, w, _ = image.shape
    h_crop = min(crop_size, h)
    w_crop = min(crop_size, w)

    mask = np.max(image, axis=-1).astype(bool).astype(int)
    top0, left0, area0 = 0, 0, 0
    for top in range(h-h_crop+1):
        for left in range(w-w_crop+1):
            # calculate overlap
            area = np.sum(mask[top:top+h_crop, left:left+w_crop])
            if area>area0:
                top0, left0 = top, left

    top, left = top0, left0
    bottom = top + crop_size
    right = left + crop_size
    image = image[top:bottom, left:right, :]
    return image


def center_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image

def random_crop(image, mask, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right]
    return image, mask


def horizontal_flip(image, mask, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
        mask = mask[:, ::-1]
    return image, mask


def vertical_flip(image, mask, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
        mask = mask[::-1, :]
    return image, mask

def scale_augmentation(image: torch.Tensor, mask, scale_range, crop_size):
    scale_size = np.random.randint(*scale_range)
    # image = imresize(image, (scale_size, scale_size))
    image = torch.from_numpy(image.copy())
    mask = torch.from_numpy(mask.copy())
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    image = torch.nn.functional.interpolate(image, size=scale_size)
    image = image.squeeze(0)
    image = image.permute(1, 2, 0).numpy()

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.nn.functional.interpolate(mask, size=scale_size).numpy()
    mask = mask.squeeze(0).squeeze(0)

    if scale_size <= crop_size[0]:
        image, mask = pad_image_and_mask(image, mask, target_size=crop_size[0], image_fill_value=0,mask_fill_value=1)
    else:
        image, mask = random_crop(image, mask, crop_size)

    return image, mask

def cutout(image_origin: torch.Tensor, mask_size, mask_value='mean'):
    # image_origin = image_origin
    image = np.copy(image_origin)
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    image[top:bottom, left:right, :].fill(mask_value)
    image = torch.from_numpy(image)
    return image


def pad_image_and_mask(image, mask, target_size, image_fill_value=0, mask_fill_value=1):
    h, w, c = image.shape
    h_pad_upper = (target_size - h) // 2
    h_pad_down = (target_size - h - h_pad_upper)

    w_pad_left = (target_size - w) // 2
    w_pad_right = (target_size - w - w_pad_left)

    ret_img = np.pad(image, ((h_pad_upper, h_pad_down), (w_pad_left, w_pad_right), (0, 0)), 'constant', constant_values=image_fill_value)
    ret_mask = np.pad(mask, ((h_pad_upper, h_pad_down), (w_pad_left, w_pad_right)), 'constant', constant_values=mask_fill_value)

    return ret_img, ret_mask

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        ## type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def crop_wsi(gather_flag=False,patch_loc=None,patch_feat=None,feat_map_size = 60,is_train=False):
    # True for cam16 (default:False)
    
    if not gather_flag:
        try:
            num_channels = 1024
            min_r = min([x[0] for x in patch_loc])
            min_c = min([x[1] for x in patch_loc])

            patch_loc = [(x[0] - min_r, x[1] - min_c) for x in patch_loc]

            max_r = max([x[0] for x in patch_loc])
            max_c = max([x[1] for x in patch_loc])
        except:
            raise ValueError(f'Error')
    else:
        # gather together patches for cam16
        num_channels = 1024
        slide_len = int(np.ceil(np.sqrt(len(patch_feat))))
        patch_loc = [(int(num//slide_len), int(num%slide_len)) for num in range(len(patch_feat))]
        max_r = max([x[0] for x in patch_loc])
        max_c = max([x[1] for x in patch_loc])

    feat_map = np.zeros((max_r+1, max_c+1, num_channels))
    #print(feat_map.shape)
    #print(feat_map.shape, image_id, self.pid_2_img_id[image_id])
    for (r_idx, c_idx), each_ob in zip(patch_loc, patch_feat):
        #if self.is_training and 'tr' in each_ob.keys():
    #     if 'tr' in each_ob.keys():
    #         c_feat = each_ob['tr']
    #         #c_feat = np.array(c_feat)
    #         select_idx = np.random.choice(c_feat.shape[0], 1).item()
    #         select_feat = c_feat[select_idx]
    #         feat_map[r_idx, c_idx, :] = select_feat.reshape(-1)
    #     else:
        select_feat = each_ob
        #select_feat = np.array(select_feat)
        feat_map[r_idx, c_idx, :] = select_feat.reshape(-1)

    # samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
    h, w = feat_map.shape[:2]
    # DTMIL
    if h > feat_map_size:
        feat_map = center_crop(feat_map, (feat_map_size, w))
    if w > feat_map_size:
        feat_map = center_crop(feat_map, (h, feat_map_size))

    mask = np.zeros(feat_map.shape[:2])
    #target_size = max([h, w, 12])
    target_size = feat_map_size
    feat_map, mask = pad_image_and_mask(feat_map, mask, target_size=target_size, image_fill_value=0, mask_fill_value=1)
    # x = np.sum(feat_map)
    #print("padding", feat_map.shape)

    if is_train:
        transform = TrFeatureMapAug()
        feat_map, mask = transform(feat_map, mask)

    # y = np.sum(feat_map)
    feat_map = torch.from_numpy(feat_map.copy()).float()
    mask = torch.from_numpy(mask.copy())

    feat_map = feat_map.permute(2, 0, 1)
    mask = mask.bool()

    return feat_map,mask

class TrFeatureMapAug:
    def __call__(self, feat, mask, p=0.5):
        feat, mask = horizontal_flip(feat, mask)
        feat, mask = vertical_flip(feat, mask)

        if np.random.rand() < p:
            scale_range_max = feat.shape[0] * 1.2
            scale_range_min = feat.shape[0] * 0.8
            crop_size = feat.shape[:2]
            scale_range = (scale_range_min, scale_range_max)
            feat, mask = scale_augmentation(feat, mask, scale_range=scale_range, crop_size=crop_size)

        if np.random.rand() < p:
            mask_size = feat.shape[0] // 8
            feat = cutout(feat, mask_size=mask_size)

        rot_k = np.random.randint(0, 4)

        feat = np.rot90(feat, k=rot_k, axes=(0, 1))
        mask = np.rot90(mask, k=rot_k, axes=(0, 1))

        return feat, mask