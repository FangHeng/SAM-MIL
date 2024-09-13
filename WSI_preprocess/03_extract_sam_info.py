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

# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def process_info(coords, seg_masks, group_info):
    '''
    Process the information of the patches and segmentation masks to identify group features and calculate the ratio of mask

    Args:
        coords: coordinates of patches
        seg_masks: segmentation masks
        group_info: group information
    '''
    # check if is_group_feat or not
    is_group_feat = [1 if np.array_equal(coord, [-1, -1]) else 0 for coord in coords]

    # initialize relative_area
    relative_area = [0 if is_feat else 1 for is_feat in is_group_feat]

    # build a dictionary to map coordinates to index
    coord_to_index = {tuple(coord): i for i, coord in enumerate(coords)}

    for seg_key, patches in group_info.items():

        seg_index = int(seg_key.split('_')[1])
        # get the area of the current group
        area = seg_masks[seg_index]['area']
        print(f'Processing {seg_key}: area = {area}')

        for patch_coord in patches:
            patch_coord_tuple = tuple(patch_coord)
            if patch_coord_tuple in coord_to_index:
                index = coord_to_index[patch_coord_tuple]

                if relative_area[index] == 1:
                    relative_area[index] = area
                elif relative_area[index] > 1:
                    relative_area[index] = min(relative_area[index], area)
                else:
                    relative_area[index] = 0

    coords_array = np.array(coords)
    is_group_feat_array = np.array(is_group_feat)
    relative_area_array = np.array(relative_area)

    return coords_array, is_group_feat_array, relative_area_array

# ----------------- main -----------------
parser = argparse.ArgumentParser(description='Identify group features and caculate the ratio of mask')
# patches h5 parameters
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')

# SAM parameters
parser.add_argument('--data_feat_h5_dir', type=str, default=None)
parser.add_argument('--data_segment_dir', type=str, default=None)
parser.add_argument('--data_group_dir', type=str, default=None)
args = parser.parse_args()


if __name__ == '__main__':
    print("########## Identify group features and caculate the ratio of mask ##########")

    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.output_dir, exist_ok=True)

    dest_files = os.listdir(os.path.join(args.output_dir))

    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.h5' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        h5_path = os.path.join(args.data_feat_h5_dir, bag_name)
        group_path = os.path.join(args.data_group_dir, bag_name)

        file = h5py.File(h5_path, "r")
        group_info = h5py.File(group_path, "r")

        coords = file['coords'][:]
        print('coordinates size: ', coords.shape)


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
        seg_masks = seg_data_list[0][1:]

        output_path = os.path.join(args.output_dir, bag_name)

        coords, is_group_feat, relative_area = process_info(coords, seg_masks, group_info)

        asset_dict = {'coords': coords, 'is_group_feat': is_group_feat, 'relative_area': relative_area}
        save_hdf5(output_path, asset_dict, mode='w')
