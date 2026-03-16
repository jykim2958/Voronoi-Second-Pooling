"""
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CSIRO BSD-3-Clause-Clear (Non-Commercial)

Copyright (c) 2025, The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.
All rights reserved. CSIRO is willing to grant you a licence to this software (Wild-Places) on the following terms, except where otherwise indicated for third party material.

CSIRO hereby grants you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute,
and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education,
or non-commercial research projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation
in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

Redistribution and use of this software in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
•	Neither the name of CSIRO nor the names of its authors may be used to endorse or promote products derived from this software without specific prior written permission of CSIRO.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR CSIRO BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors and/or CSIRO are under no obligation to provide either maintenance services, update services,
notices of latent defects, or corrections of defects with regard to the Software.

The Software may contain third party material obtained by CSIRO under licence.  Your rights to such material as part of the Software under this agreement
is subject to any separate licence terms identified by CSIRO as part of the Software release - including as part of the Supplementary Licence, or as a separate file.
Those third party licence terms may require you to download the relevant software from a third party site, or may mean that the third
party licensor (and not CSIRO) grants you a licence directly for those components of the Software. It is your responsibility to ensure that
you have the necessary rights to such third party material.


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""
import os 
import sys
sys.path.append(f"{os.getcwd()}")
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KDTree 
import pickle 
import argparse 
from tqdm import tqdm 
from datasets.wildplaces.utils import TrainingTupleWP, load_csv, check_in_test_set
from datasets.wildplaces.utils import P1, P2, P3, P4, P5, P6
from datasets.wildplaces.utils import B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12
import matplotlib.pyplot as plt 


# Test set boundaries 
POLY_VENMAN = [P1,P2,P3]
EXCLUDE_VENMAN = [B1, B2, B3, B4, B5, B6]

POLY_KARAWATHA = [P4,P5,P6]
EXCLUDE_KARAWATHA = [B7, B8, B9, B10, B11, B12]

_OFFSET = 2000

def construct_query_dict(df_centroids, save_path, ind_nn_r, ind_r_r):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['easting', 'northing']])
    ind_nn = tree.query_radius(df_centroids[['easting', 'northing']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['easting', 'northing']], r=ind_r_r)
    queries = {}
    for anchor_ndx in tqdm(range(len(ind_nn))):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['easting', 'northing']])
        pose = np.array(df_centroids.iloc[anchor_ndx][['x','y','z','qx','qy','qz','qw']])
        query = df_centroids.iloc[anchor_ndx]["filename"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        timestamp = float(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: str, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTupleWP(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                              positives=positives, non_negatives=non_negatives, position=anchor_pos,
                                              pose = pose)

    file_path = save_path
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training dataset')
    parser.add_argument('--dataset_root', type = str, required = True, help = 'Dataset root folder')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save training pickles to')

    parser.add_argument('--csv_filename', type = str, default = 'poses_aligned.csv', help = 'Name of CSV containing ground truth poses')
    parser.add_argument('--cloud_folder', type = str, default = 'Clouds_downsampled', help = 'Name of folder containing point cloud frames')

    parser.add_argument('--pos_thresh', type = float, default = 3, help  = 'Threshold to sample positives within')
    parser.add_argument('--neg_thresh', type = float, default = 50, help = 'Threshold to sample negative within')

    parser.add_argument('--vis_splits', action = 'store_true', default = False, help = 'View training splits using matplotlib')
    args = parser.parse_args()
    
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    print(f'Dataset root: {args.dataset_root}')
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    ##### Venman #####
    venman_all_coords = []
    venman_all_c = []

    # Get training runs, i.e. first two
    base = os.path.join(args.dataset_root, 'Venman')
    train_folders = sorted(os.listdir(base))[:2]

    df_train_venman = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])
    df_test_venman = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])

    for idx, folder in enumerate(train_folders):
        df_locations = load_csv(os.path.join(base, folder, args.csv_filename), rel_cloud_path = os.path.join('Venman', folder, args.cloud_folder))

        for index, row in tqdm(df_locations.iterrows(), desc = f'Venman {idx+1} / {len(train_folders)}', total = len(df_locations)):
            venman_all_coords.append(row[['easting', 'northing']])
            row_split = check_in_test_set(row['easting'], row['northing'], test_polygons = POLY_VENMAN, exclude_polygons = EXCLUDE_VENMAN)
            if  row_split == 'test':
                df_test_venman.loc[len(df_test_venman)] = row 
                venman_all_c.append([1,0,0])
            elif row_split == 'train':
                df_train_venman.loc[len(df_train_venman)] = row
                venman_all_c.append([0,0,1])
            elif row_split == 'buffer':
                venman_all_c.append([1, 165 / 255, 0])



    ##### Karawatha #####
    karawatha_all_coords = []
    karawatha_all_c = []

    # Get training runs, i.e. first two
    base = os.path.join(args.dataset_root, 'Karawatha')
    train_folders = sorted(os.listdir(base))[:2]

    df_train_karawatha = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])
    df_test_karawatha = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])

    for idx, folder in enumerate(train_folders):
        df_locations = load_csv(os.path.join(base, folder, args.csv_filename), rel_cloud_path = os.path.join('Karawatha', folder, args.cloud_folder))

        for index, row in tqdm(df_locations.iterrows(), desc = f'Karawatha {idx+1} / {len(train_folders)}', total = len(df_locations)):
            karawatha_all_coords.append(row[['easting', 'northing']])
            row_split = check_in_test_set(row['easting'], row['northing'], test_polygons = POLY_KARAWATHA, exclude_polygons = EXCLUDE_KARAWATHA)
            if  row_split == 'test':
                df_test_karawatha.loc[len(df_test_karawatha)] = row 
                karawatha_all_c.append([1,0,0])
            elif row_split == 'train':
                df_train_karawatha.loc[len(df_train_karawatha)] = row
                karawatha_all_c.append([0,0,1])
            elif row_split == 'buffer':
                karawatha_all_c.append([1, 165 / 255, 0])

    # Offset Karawatha by a large number to stop the ground truth maps from overlapping
    df_train_karawatha[['x', 'easting']] = df_train_karawatha[['x', 'easting']] + _OFFSET
    df_test_karawatha[['x', 'easting']] = df_test_karawatha[['x', 'easting']] + _OFFSET
    # karawatha_all_coords = [[x[0] + _OFFSET, x[1]] for x in karawatha_all_coords]

    #### Combined ####
    df_train_both = pd.concat([df_train_venman, df_train_karawatha], ignore_index = True)
    df_test_both = pd.concat([df_test_venman, df_test_karawatha], ignore_index = True)

    all_coords = np.array(venman_all_coords + karawatha_all_coords)
    all_c = np.array(venman_all_c + karawatha_all_c)


    ### Summary ###
    print('\nVenman: ')
    total_submaps = len(venman_all_coords)
    print(f"\tNumber of training submaps: {len(df_train_venman)} ({len(df_train_venman) / total_submaps})%")
    print(f"\tNumber of non-disjoint test submaps: {len(df_test_venman)} ({len(df_test_venman) / total_submaps})%")
    print(f"\tNumber of excluded submaps in bufferzone {total_submaps - len(df_train_venman) - len(df_test_venman)} ({(total_submaps - len(df_train_venman) - len(df_test_venman)) * 100 / total_submaps}%)")

    print('\nKarawatha: ')
    total_submaps = len(karawatha_all_coords)
    print(f"\tNumber of training submaps: {len(df_train_karawatha)} ({len(df_train_karawatha) / total_submaps})%")
    print(f"\tNumber of non-disjoint test submaps: {len(df_test_karawatha)} ({len(df_test_karawatha) / total_submaps})%")
    print(f"\tNumber of excluded submaps in bufferzone {total_submaps - len(df_train_karawatha) - len(df_test_karawatha)} ({(total_submaps - len(df_train_karawatha) - len(df_test_karawatha)) * 100 / total_submaps}%)")

    print('\nBoth: ')
    total_submaps = len(all_coords)
    print(f"\tNumber of training submaps: {len(df_train_both)} ({len(df_train_both) / total_submaps})%")
    print(f"\tNumber of non-disjoint test submaps: {len(df_test_both)} ({len(df_test_both) / total_submaps})%")
    print(f"\tNumber of excluded submaps in bufferzone {total_submaps - len(df_train_both) - len(df_test_both)} ({(total_submaps - len(df_train_both) - len(df_test_both)) * 100 / total_submaps}%)")


    ### Vis if selected ###
    if args.vis_splits == True:
        venman_all_coords = np.array(venman_all_coords)
        karawatha_all_coords = np.array(karawatha_all_coords)

        fig, (ax2, ax3) = plt.subplots(1, 2)
        # ax1.scatter(all_coords[:,0], all_coords[:,1], c = all_c)

        ax2.scatter(venman_all_coords[:,0], venman_all_coords[:,1], c = venman_all_c)
        for poly in POLY_VENMAN:
            ax2.plot(*poly.exterior.xy)
        
        ax3.scatter(karawatha_all_coords[:,0], karawatha_all_coords[:,1], c = karawatha_all_c)
        for poly in POLY_KARAWATHA:
            ax3.plot(*poly.exterior.xy)
        plt.show()

    construct_query_dict(df_train_both, os.path.join(args.save_folder, f"training_wild-places.pickle"), ind_nn_r = args.pos_thresh, ind_r_r = args.neg_thresh)
    construct_query_dict(df_test_both, os.path.join(args.save_folder, f"testing_wild-places.pickle"), ind_nn_r = args.pos_thresh, ind_r_r = args.neg_thresh)
