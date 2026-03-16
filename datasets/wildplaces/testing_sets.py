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

def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name, save_dir, eval_thresh):
    database_trees = []
    test_trees = []

    # Create KD Trees
    for folder in tqdm(folders):
        # Create dataframes 
        df_database = pd.DataFrame(columns=['file', 'easting', 'northing'])
        df_test = pd.DataFrame(columns=['file', 'easting', 'northing'])

        # Load dataframe 
        df_locations = load_csv(os.path.join(base_path, runs_folder, folder, filename), os.path.join(runs_folder, folder, pointcloud_fols))
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['easting'], row['northing'], p, []) == 'test':
                df_test.loc[len(df_test)] = row 
            df_database.loc[len(df_database)] = row 

        database_tree = KDTree(df_database[['easting', 'northing']])
        test_tree = KDTree(df_test[['easting', 'northing']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in tqdm(folders):
        database = {}
        test = {}
        df_locations = load_csv(os.path.join(base_path, runs_folder, folder, filename), os.path.join(runs_folder, folder, pointcloud_fols))
        for index, row in df_locations.iterrows():
            pose = np.array(row[['x','y','z','qx','qy','qz','qw']])        
            row['timestamp'] = float(os.path.basename(row['filename'].replace('.pcd', ''))) 
            if check_in_test_set(row['easting'], row['northing'], p, []) == 'test':
                test[len(test.keys())] = {'query': row['filename'], 'northing': row['northing'], 'easting': row['easting'], 'pose': pose, 'timestamp': row['timestamp']}
            database[len(database.keys())] = {'query': row['filename'], 'northing': row['northing'],
                                              'easting': row['easting'], 'pose': pose, 'timestamp': row['timestamp']}

            # Output to file for in-run evaluation
            single_run_output_name = os.path.basename(folder) + '.pickle'
        output_to_file(database, save_dir, single_run_output_name)
            

        database_sets.append(database)
        test_sets.append(test)

    for i in tqdm(range(len(database_sets))):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["easting"], test_sets[j][key]["northing"]]])
                index = tree.query_radius(coor, r=eval_thresh)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    print(f'{output_name}: Query / Database Size {sum([len(x) for x in test_sets])} / {sum([len(x) for x in database_sets])}')

    output_to_file(database_sets, save_dir, output_name + f'_evaluation_database.pickle')
    output_to_file(test_sets, save_dir, output_name + f'_evaluation_query.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate testing dataset')
    parser.add_argument('--dataset_root', type = str, required = True, help = 'Dataset root folder')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save training pickles to')

    parser.add_argument('--csv_filename', type = str, default = 'poses_aligned.csv', help = 'Name of CSV containing ground truth poses')
    parser.add_argument('--cloud_folder', type = str, default = 'Clouds_downsampled', help = 'Name of folder containing point cloud frames')

    parser.add_argument('--eval_thresh', type = float, default = 3, help  = 'Threshold for correct retrieval during eval') 
    args = parser.parse_args()

    print('Dataset root: {}'.format(args.dataset_root))

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    # For Venman
    folders = sorted(os.listdir(os.path.join(args.dataset_root, 'Venman')))
    print('Venman')
    construct_query_and_database_sets(
        base_path = args.dataset_root,
        runs_folder = 'Venman',
        folders = folders,
        pointcloud_fols = args.cloud_folder,
        filename = args.csv_filename,
        p = [P1,P2,P3],
        output_name = 'Venman',
        save_dir = args.save_folder,
        eval_thresh = args.eval_thresh
    )

    # For Karawatha
    folders = sorted(os.listdir(os.path.join(args.dataset_root, 'Karawatha')))
    print('Karawatha')
    construct_query_and_database_sets(
        base_path = args.dataset_root,
        runs_folder = 'Karawatha',
        folders = folders,
        pointcloud_fols = args.cloud_folder,
        filename = args.csv_filename,
        p = [P4,P5,P6],
        output_name = 'Karawatha',
        save_dir = args.save_folder,
        eval_thresh = args.eval_thresh
    )

