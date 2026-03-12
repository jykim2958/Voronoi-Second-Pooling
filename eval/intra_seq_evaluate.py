# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
import os
import sys
import time
# sys.path.append("/home/jykim/workspace/MinkLoc3Dv2")
sys.path.append(f"{os.getcwd()}")
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import argparse
import torch
import MinkowskiEngine as ME
import random
import tqdm
import copy

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader, WildPlacesPointCloudLoader


def query_to_timestamp(query):
    base = os.path.basename(query)
    timestamp = float(base.replace('.pcd', ''))
    return timestamp

def euclidean_dist(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()


def evaluate(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False):
    # Run evaluation on all eval datasets

    assert params.dataset_type == 'wildplaces'
    eval_database_files = ['V-03.pickle', 'V-04.pickle', 'K-03.pickle', 'K-04.pickle']
    # eval_database_files = ['V-04.pickle']
    stats = {}
    for database_file in eval_database_files:
        # Extract location name from query and database files
        # NOTE Intra-sequence
        location_name = database_file.split('.')[0]
        
        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        # p = os.path.join(params.dataset_folder, query_file)
        # with open(p, 'rb') as f:
        #     query_sets = pickle.load(f)

        temp, ROC = evaluate_dataset(model, device, params, database_sets, log=log, show_progress=show_progress)
        stats[location_name] = temp
        np.save(f'{location_name}_ROC.npy', ROC)

    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, log: bool = False,
                     show_progress: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []
    map_at_r = []
    mrr = []

    model.eval()

    # NOTE Intra-sequences
    # database_embeddings.append(get_latent_vectors(model, database_sets, device, params))
    st = time.time()
    embeddings = get_latent_vectors(model, database_sets, device, params)
    ed = time.time()
    stats, ROC = eval_singlesession(database_sets, embeddings)
    get_latent_speed = (ed - st) / stats['Sequence Length']
    print(f'Latent Computation Speed: {round(get_latent_speed, 4)}')
    return stats, ROC


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
    for i, elem_ndx in enumerate(tqdm.tqdm(set, desc='Latent Vectors', leave=False)):
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


def eval_singlesession(database, embeddings):

    TIME_THRESH = 600 # 300
    WORLD_THRESH = 3 # 5
    timestamps = [query_to_timestamp(database[k]['query']) for k in range(len(database.keys()))]
    coords = np.array([[database[k]['easting'], database[k]['northing']] for k in range(len(database.keys()))])
    start_time = timestamps[0]

    # Thresholds, other trackers
    thresholds = np.linspace(0, 1, 1000) # NOTE
    # thresholds = np.linspace(0, 7, 7000)
    num_thresholds = len(thresholds)

    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    # Get similarity function 
    dist_func = euclidean_dist

    num_revisits = 0
    num_correct_loc = 0

    st = time.time()
    for query_idx in tqdm.tqdm(range(len(database.keys())), desc = 'Evaluating Embeddings'):
        q_embedding = embeddings[query_idx]
        q_timestamp = timestamps[query_idx]
        q_coord = coords[query_idx]

        # Exit if time elapsed since start is less than time threshold 
        if (q_timestamp - start_time - TIME_THRESH) < 0:
            continue 

        # Build retrieval database 
        tt = next(x[0] for x in enumerate(timestamps) if x[1] > (q_timestamp - TIME_THRESH))
        seen_embeddings = embeddings[:tt+1]
        seen_coords = coords[:tt+1]

        # Get distances in feature space and world 
        dist_seen_embedding = dist_func(q_embedding, seen_embeddings)
        dist_seen_world = euclidean_dist(q_coord, seen_coords)

        # Check if re-visit 
        if np.any(dist_seen_world < WORLD_THRESH):
            revisit = True 
            num_revisits += 1 
        else:
            revisit = False 

        # Get top-1 candidate and distances in real world, embedding space 
        top1_idx = np.argmin(dist_seen_embedding)
        top1_embed_dist = dist_seen_embedding[top1_idx]
        top1_world_dist = dist_seen_world[top1_idx]

        if top1_world_dist < WORLD_THRESH:
            num_correct_loc += 1 
        
        # Evaluate top-1 candidate 
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if top1_embed_dist < threshold: # Positive Prediction
                if top1_world_dist < WORLD_THRESH:
                    num_true_positive[thresh_idx] += 1
                else:
                    num_false_positive[thresh_idx] += 1
            else: # Negative Prediction
                if not revisit:
                    num_true_negative[thresh_idx] += 1
                else:
                    num_false_negative[thresh_idx] += 1
    ed = time.time()

    # Find F1Max and Recall@1 
    recall_1 = num_correct_loc / num_revisits

    F1max = 0.0
    best_thresh_idx = 0
    ROC = np.zeros((num_thresholds, 2)) # First dim is TPR, Second dim is FPR
    for thresh_idx in range(num_thresholds):
        nTruePositive = num_true_positive[thresh_idx]
        nFalsePositive = num_false_positive[thresh_idx]
        nTrueNegative = num_true_negative[thresh_idx]
        nFalseNegative = num_false_negative[thresh_idx]

        Precision = 0.0
        Recall = 0.0
        F1 = 0.0

        FPR = nFalsePositive / (nFalsePositive + nTrueNegative)
        TPR = nTruePositive / (nTruePositive + nFalseNegative)
        ROC[thresh_idx] = np.array([FPR, TPR])

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1
            best_thresh_idx = thresh_idx

    print(f'Evaluation Speed: {round((ed - st) / len(embeddings), 4)}')
    print(f'Best Threshold Idx: {best_thresh_idx}')
    return {'F1max': F1max, 'Recall@1': recall_1, 'Sequence Length': len(embeddings), 'Num. Revisits': num_revisits, 'Num. Correct Locations': num_correct_loc}, ROC


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        # t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        # print(t.format(stats[database_name]['ave_one_percent_recall']))
        # print(stats[database_name]['ave_recall'])
        t = 'F1 Max: {:.4%}   Recall@1: {:.4%}   Seq Length: {}   Num. Revisited: {}   Num. Correct Locs: {}'
        print(t.format(stats[database_name]['F1max'], stats[database_name]['Recall@1'], stats[database_name]['Sequence Length'], stats[database_name]['Num. Revisits'], stats[database_name]['Num. Correct Locations']))


# def pnv_write_eval_stats(file_name, prefix, stats):
#     s = prefix
#     ave_1p_recall_l = []
#     ave_recall_l = []
#     # Print results on the final model
#     with open(file_name, "a") as f:
#         for ds in stats:
#             ave_1p_recall = stats[ds]['ave_one_percent_recall']
#             ave_1p_recall_l.append(ave_1p_recall)
#             ave_recall = stats[ds]['ave_recall'][0]
#             ave_recall_l.append(ave_recall)
#             s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)
# 
#         mean_1p_recall = np.mean(ave_1p_recall_l)
#         mean_recall = np.mean(ave_recall_l)
#         s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
#         f.write(s)


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

    # # Save results to the text file
    # model_params_name = os.path.split(params.model_params.model_params_path)[1]
    # config_name = os.path.split(params.params_path)[1]
    # model_name = os.path.split(args.weights)[1]
    # model_name = os.path.splitext(model_name)[0]
    # prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    # pnv_write_eval_stats("pnv_experiment_results.txt", prefix, stats)

