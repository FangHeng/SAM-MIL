"""
This code is adapted from the following open-source project:
    CLAM: Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images.
    - GitHub Repository: https://github.com/mahmoodlab/CLAM

Description:
    Our code adds SAM segmentation in addition to the patching operation, where the SAM implementation references the original repository implementation. The processes are merged into one complete process.

Note:
    - Parts of the code may have been modified or extended to suit specific requirements.
    - Refer to the original repository for detailed documentation and licensing information.
"""


import time
import h5py
import argparse
import os

import torch
import numpy as np
import pickle
import openslide

from utils.file_utils import save_hdf5
from datasets.dataset_h5 import Dataset_All_Bags



def is_patch_in_segmentation(h5_path, seg_path, coords, feature, seg_masks, seg_params, adjusted_patch_size, downsample_factor):
    """
    check if the patch is in the segmentation mask and extract group features
    and save patches' seg info

    args:
        h5_path: h5 file path
        seg_path: segmentation file path
        coords: coordinates of patches
        feature: features of patches
        seg_masks: segmentation masks
        seg_params: segmentation parameters
        adjusted_patch_size: adjusted patch size
        downsample_factor: downsample factor
    """
    # print("processing segmentation masks and extracting group features")
    start_time = time.time()
    mode = 'w'
    # loop through all segmentation masks
    for seg_index, seg_mask in enumerate(seg_masks):
        in_seg_coords = []
        in_seg_features = []

        # loop through all patches
        for coord, feat in zip(coords, feature):
            # adjust the coordinates
            adjusted_coord = ((coord - np.array([seg_params['start_x'], seg_params['start_y']])) / downsample_factor).round().astype(int)

            # check if the patch is in the bbox
            bbox = seg_mask['bbox']
            if not (bbox[0] <= adjusted_coord[0] < bbox[0] + bbox[2] and bbox[1] <= adjusted_coord[1] < bbox[1] + bbox[3]):
                # print(f'patch not in bbox, seg index: {seg_index}')
                continue

            # check if the patch is in the segmentation mask
            mask = seg_mask['segmentation']
            patch_x, patch_y = adjusted_coord.astype(int)
            if np.any(mask[patch_y:patch_y + adjusted_patch_size, patch_x:patch_x + adjusted_patch_size]):
                in_seg_coords.append(coord)
                in_seg_features.append(feat)

        # if no patch in the segmentation mask
        if not in_seg_coords:
            # print(f'no patch in seg index: {seg_index}')
            continue

        # extract group features
        group_feature = np.expand_dims(np.mean(in_seg_features, axis=0), axis=0)

        # save group features
        asset_dict = {'features': group_feature, 'coords': np.array([[-1, -1]])}
        save_hdf5(h5_path, asset_dict, attr_dict=None, mode='a')
        # print(f'save group feature in seg index: {seg_index} to {h5_path}')

        # update seg info
        with h5py.File(seg_path, mode) as seg_file:
            ds_name = f'seg_{seg_index}'
            seg_file.create_dataset(ds_name, data=np.array(in_seg_coords))
            seg_file[ds_name].attrs['seg_index'] = seg_index
        mode = 'a'

    end_time = time.time()
    print(f'processing segmentation masks and extracting group features took {end_time - start_time} s')

    return seg_path


# ----------------- main -----------------
parser = argparse.ArgumentParser(description='h5 Group Feature Extraction')
# patches h5 parameters
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')

# SAM parameters
parser.add_argument('--use_sam', default=False, action='store_true')
parser.add_argument('--data_feat_h5_dir', type=str, default=None)
parser.add_argument('--data_segment_dir', type=str, default=None)
# parser.add_argument('--group_features', default=False, action='store_true')
parser.add_argument('--patch_size', type=int, default=256,
                    help='patch size for the patches, only for SAM analysis')
# parser.add_argument('--'
args = parser.parse_args()


if __name__ == '__main__':
    print("########## extract features from h5 file ##########")

    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'seg_files'), exist_ok=True)

    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))


    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        h5_path = os.path.join(args.data_feat_h5_dir, bag_name)
        seg_output_path = os.path.join(args.feat_dir, 'seg_files', bag_name)

        wsi = openslide.open_slide(slide_file_path)

        file = h5py.File(h5_path, "r")

        features = file['features'][:]
        coords = file['coords'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', coords.shape)

        # SAM group features
        if args.use_sam:
            seg_name = slide_id + '.pkl'
            seg_file_path = os.path.join(args.data_segment_dir, seg_name)

            seg_data_list = []
            with open(seg_file_path, 'rb') as file:
                while True:
                    try:
                        data = pickle.load(file)
                        seg_data_list.append(data)
                    except EOFError:
                        break
            seg_params = seg_data_list[0][0]
            seg_masks = seg_data_list[0][1:]

            downsample_factor = int(wsi.level_downsamples[seg_params['vis_level']])
            # print(f'vis_level: {seg_params["vis_level"]}, downsample_factor: {downsample_factor}, downsample_ratio: {seg_params["downsample_ratio"]}')
            downsample_factor = downsample_factor * seg_params['downsample_ratio']
            print(f'downsample_factor: {downsample_factor}x')
            adjusted_patch_size = args.patch_size // downsample_factor
            print(f'adjusted_patch_size: {adjusted_patch_size}')

            output_seg_path = is_patch_in_segmentation(h5_path, seg_output_path, coords, features,
                                                       seg_masks, seg_params, adjusted_patch_size, downsample_factor)
            print(f'extracted features by SAM: {output_seg_path}')

            file = h5py.File(h5_path, "r")
            features = file['features'][:]
            coords = file['coords'][:]

        else:
            print('no SAM, just save original features')

            # find the patches that index is -1
            indices_to_remove = np.all(coords == [-1, -1], axis=1)

            # remove the patches
            features = features[~indices_to_remove]
            coords = coords[~indices_to_remove]

        print('features size: ', features.shape)
        print('coordinates size: ', coords.shape)

        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))