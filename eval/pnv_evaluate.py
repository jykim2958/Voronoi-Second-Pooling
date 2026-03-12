'''
Komorowski, Jacek. "Improving point cloud based place recognition with ranking-based loss and large batch training."
26th International Conference on Patter Recognition (2022).
This implementation is adopted from https://github.com/jac99/MinkLoc3Dv2

MIT License

Copyright (c) 2022 jac99

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
import os
import sys
sys.path.append(f"{os.getcwd()}")
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import argparse
import torch
import MinkowskiEngine as ME
import random
import tqdm

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader, WildPlacesPointCloudLoader


def evaluate(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False):
    # Run evaluation on all eval datasets

    if params.dataset_type == 'oxford':
        eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
                               'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']
        eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
                            'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']
    elif params.dataset_type == 'wildplaces':
        eval_database_files = ['Venman_evaluation_database.pickle', 'Karawatha_evaluation_database.pickle']
        eval_query_files = ['Venman_evaluation_query.pickle', 'Karawatha_evaluation_query.pickle']
    else:
        raise NotImplementedError

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        # NOTE Inter-sequence
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)
        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params, database_sets, query_sets, log=log, show_progress=show_progress)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets, log: bool = False,
                     show_progress: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []
    map_at_r = []
    mrr = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    # NOTE Inter-sequences
    for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        database_embeddings.append(get_latent_vectors(model, set, device, params))

    # Dist = compute_total_covariance(database_embeddings)
    Dist = 0.

    for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        query_embeddings.append(get_latent_vectors(model, set, device, params))

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
            # pair_recall, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
            #                                    database_sets, log=log)
            pair_recall, pair_opr, pair_mapr, pair_mrr = get_recall_n_precision(i, j, database_embeddings, query_embeddings, query_sets,
                                                                      database_sets, log=log)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            map_at_r = map_at_r + pair_mapr
            mrr.append(pair_mrr)

    # ave_recall = recall / count
    # ave_one_percent_recall = np.mean(one_percent_recall)
    # stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    # map_at_r = np.mean(map_at_r)
    map_at_r = np.mean(np.stack(map_at_r, axis=0), axis=0) * 100
    mrr = np.mean(mrr)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall, 'map_at_10': map_at_r, 'mrr': mrr, 'cov_dist_I': Dist}
    return stats


def get_latent_vectors(model, set, device, params: TrainingParams):
    # Adapted from original PointNetVLAD code

    if params.debug:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    if params.dataset_type == 'oxford':
        pc_loader = PNVPointCloudLoader()
    elif params.dataset_type == 'wildplaces':
        pc_loader = WildPlacesPointCloudLoader()
    else:
        raise NotImplementedError

    model.eval()
    embeddings = None
    for i, elem_ndx in enumerate(set):
        pc_file_path = os.path.join(params.dataset_folder, set[elem_ndx]["query"])
        pc = pc_loader(pc_file_path)
        pc = torch.tensor(pc)

        embedding = compute_embedding(model, pc, device, params)
        if embeddings is None:
            embeddings = np.zeros((len(set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings


def compute_embedding(model, pc, device, params: TrainingParams):
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        # Compute global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        if log:
            # Log false positives (returned as the first element) for Oxford dataset
            # Check if there's a false positive returned as the first element
            if query_details['query'][:6] == 'oxford' and indices[0][0] not in true_neighbors:
                fp_ndx = indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors:
                        closest_pos_ndx = indices[0][k]
                        tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[0][k]
                        tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
                                                (query_details['easting'] - tp['easting']) ** 2)
                        break

                with open("log_fp.txt", "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            if query_details['query'][:6] == 'oxford':
                # Save details of 5 best matches for later visualization for 1% of queries
                s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
                for k in range(min(len(indices[0]), 5)):
                    is_match = indices[0][k] in true_neighbors
                    e_ndx = indices[0][k]
                    e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                    e_emb_dist = distances[0][k]
                    world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
                                         (query_details['easting'] - e['easting']) ** 2)
                    s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
                s += '\n'
                out_file_name = "log_search_results.txt"
                with open(out_file_name, "a") as f:
                    f.write(s)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, one_percent_recall


def get_recall_n_precision(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors
    R = 10
    map_at_r = []
    recall_idx = []

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        if log:
            # Log false positives (returned as the first element) for Oxford dataset
            # Check if there's a false positive returned as the first element
            if query_details['query'][:6] == 'oxford' and indices[0][0] not in true_neighbors:
                fp_ndx = indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors:
                        closest_pos_ndx = indices[0][k]
                        tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[0][k]
                        tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
                                                (query_details['easting'] - tp['easting']) ** 2)
                        break

                with open("log_fp.txt", "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            if query_details['query'][:6] == 'oxford':
                # Save details of 5 best matches for later visualization for 1% of queries
                s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
                for k in range(min(len(indices[0]), 5)):
                    is_match = indices[0][k] in true_neighbors
                    e_ndx = indices[0][k]
                    e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                    e_emb_dist = distances[0][k]
                    world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
                                         (query_details['easting'] - e['easting']) ** 2)
                    s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
                s += '\n'
                out_file_name = "log_search_results.txt"
                with open(out_file_name, "a") as f:
                    f.write(s)
        # Recall@R & MRR
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                recall_idx.append(j+1)
                break

        # MAP@R
        # per_query_precision = np.zeros(R)
        # per_query_mask = np.zeros(R)
        # denom = np.arange(R) + 1
        # for j in range(R):
        #     if indices[0][j] in true_neighbors:
        #         per_query_precision[j] += 1
        #         per_query_mask[j] += 1
        # per_query_precision = np.cumsum(per_query_precision) * per_query_mask / denom
        # if np.sum(per_query_mask) == 0:
        #     map_at_r.append(0)
        # else:
        #     map_at_r.append(np.sum(per_query_precision) / np.sum(per_query_mask))
        per_query_precision = np.zeros((R, R))
        per_query_mask = np.zeros((R, R))
        denom = np.expand_dims(np.arange(R) + 1, axis=0) # (1, R)
        for j in range(R):
            if indices[0][j] in true_neighbors:
                per_query_precision[:, j] += 1
                per_query_mask[:, j] += 1
        for j in range(R):
            per_query_mask[j, j+1:] = 0
        per_query_precision = np.cumsum(per_query_precision, axis=1) * per_query_mask / denom
        rel_at_r = np.sum(per_query_mask, axis=1)
        # ap_at_r = np.where(rel_at_r > 0, np.sum(per_query_precision, axis=1) / rel_at_r, np.zeros(R))
        ap_at_r = np.empty_like(rel_at_r)
        mask = (rel_at_r > 0)
        ap_at_r[mask] = np.sum(per_query_precision, axis=-1)[mask] / rel_at_r[mask]
        ap_at_r[~mask] = 0
        map_at_r.append(ap_at_r)
        # Recall@1%
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    mrr = np.mean(1 / np.array(recall_idx))*100
    return recall, one_percent_recall, map_at_r, mrr


def compute_total_covariance(embeddings):

    M = 16; C = 16
    embeddings = np.vstack(embeddings)
    num_obs = embeddings.shape[0]
    assert M*C == embeddings.shape[1]
    embeddings = embeddings.reshape(num_obs, M, C)
    dist = 0.
    for i in range(M):
        x = embeddings[:, i, :]
        mu = np.mean(x, axis=0, keepdims=True)
        x = x - mu
        cov = np.cov(x, rowvar=False) * (M**2)
        I = np.eye(C)
        dist += np.linalg.norm(cov - I).item()
    return dist / M


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        # t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        # print(t.format(stats[database_name]['ave_one_percent_recall']))
        # print(stats[database_name]['ave_recall'])
        # t = 'Cov-Dist-to-I: {:.6f}   MAP@10: {:.4%}   MRR: {:.4f}   Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        # print(t.format(stats[database_name]['cov_dist_I'], stats[database_name]['map_at_10'], stats[database_name]['mrr'], stats[database_name]['ave_one_percent_recall']))
        t = 'Cov-Dist-to-I: {:.6f}   MRR: {:.4f}   Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['cov_dist_I'], stats[database_name]['mrr'], stats[database_name]['ave_one_percent_recall']))
        print(stats[database_name]['ave_recall'])
        print('MAP @N:')
        print(stats[database_name]['map_at_10'])


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true')
    parser.set_defaults(log=False)

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    print('')

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params.model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    stats = evaluate(model, device, params, args.log, show_progress=True)
    print_eval_stats(stats)

    # Save results to the text file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(args.weights)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    pnv_write_eval_stats("pnv_experiment_results.txt", prefix, stats)

