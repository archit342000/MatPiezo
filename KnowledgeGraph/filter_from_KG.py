import os
import csv
import argparse
import pickle
import networkx as nx
from tqdm import tqdm
import numpy as np
from mp_api.client.routes import PiezoRester

def create_KG(fingerprints_path, dist_threshold, e_ij_max_threshold=None):

    base_dir = os.path.dirname(fingerprints_path)    

    if e_ij_max_threshold is not None:
        file_name = os.path.join(base_dir, f'KG_{dist_threshold}_{e_ij_max_threshold}.pkl')
    else:
        file_name = os.path.join(base_dir, f'KG_{dist_threshold}.pkl')

    if os.path.exists(file_name):
        print('File already exists!')
        return

    # Load the dictionary
    fps = np.load(fingerprints_path, allow_pickle=True).item()

    # Create the graph
    G = nx.Graph()

    ids = list(fps.keys())

    if e_ij_max_threshold is not None:

        pkl_file = os.path.join(base_dir, base_dir + 'eij.pkl')
        with open(pkl_file, 'rb') as f:
            piezo_data = pickle.load(f)

        # Get list of piezo_ids
        piezo_ids = list(piezo_data.keys())
        
        # Filter piezo_ids based on e_ij_max
        piezo_ids = [piezo_id for piezo_id in piezo_ids if piezo_data[piezo_id] >= e_ij_max_threshold]

        # Get intersection of ids and piezo_ids
        ids = list(set(ids).intersection(set(piezo_ids)))

    # Add the nodes
    print('Adding nodes...')
    for i in tqdm(range(len(ids))):
        G.add_node(ids[i], fp=fps[ids[i]])

    # Add the edges
    print('Adding edges...')
    for i in tqdm(range(len(ids))):
        for j in range(i+1, len(ids)):
            
            id1 = ids[i]
            id2 = ids[j]
            fp1 = fps[id1]
            fp2 = fps[id2]
            dist = np.linalg.norm(fp1 - fp2)

            if dist > dist_threshold:
                continue

            G.add_edge(id1, id2, distance=dist)

    print('Done! Saving...')

    with open(file_name, 'wb') as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter from KG')
    parser.add_argument('--src_fps', type=str, required=True, help='source KG file')
    parser.add_argument('--tgt_fps', type=str, required=True, help='target KG file')
    parser.add_argument('--src_thresh', type=float, default=0.9, help='source KG threshold')
    parser.add_argument('--tgt_thresh', type=float, default=0.9, help='target KG threshold')
    parser.add_argument('--merge_thresh', type=float, default=0.9, help='merge KG threshold')
    parser.add_argument('--e_ij_max_thresh', type=float, default=1.0, help='e_ij_max threshold')
    parser.add_argument('--n_hops', type=int, default=2, help='number of hops')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    args = parser.parse_args()

    # Get arguments
    src_fps = args.src_fps
    tgt_fps = args.tgt_fps
    src_thresh = args.src_thresh
    tgt_thresh = args.tgt_thresh
    merge_thresh = args.merge_thresh
    e_ij_max_thresh = args.e_ij_max_thresh
    n_hops = args.n_hops
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create source KG
    create_KG(src_fps, src_thresh, e_ij_max_thresh)

    # Create target KG
    create_KG(tgt_fps, tgt_thresh)

    # Source KG file
    src_KG_file = os.path.join(os.path.dirname(src_fps), f'KG_{src_thresh}_{e_ij_max_thresh}.pkl')

    # Target KG file
    tgt_KG_file = os.path.join(os.path.dirname(tgt_fps), f'KG_{tgt_thresh}.pkl')

    # Load source KG file
    print('Loading source KG file...')
    with open(src_KG_file, 'rb') as f:
        src_KG = pickle.load(f)

    # Load target KG file
    print('Loading target KG file...')
    with open(tgt_KG_file, 'rb') as f:
        tgt_KG = pickle.load(f)

    # Source KG nodes
    src_KG_nodes = list(src_KG.nodes)

    # Target KG nodes
    tgt_KG_nodes = list(tgt_KG.nodes)

    # Get list of target KG nodes that are not in source KG
    tgt_KG_nodes_not_in_src_KG = [node for node in tgt_KG_nodes if node not in src_KG_nodes]

    # List of output nodes
    output_nodes = [node for node in tgt_KG_nodes if node in src_KG_nodes]

    # Merge the two KGs
    print('Merging the two KGs...')
    for node_src in tqdm(src_KG_nodes):
        for node_tgt in tqdm(tgt_KG_nodes_not_in_src_KG):
            fp_src = src_KG.nodes[node_src]['fp']
            fp_tgt = tgt_KG.nodes[node_tgt]['fp']
            dist = np.linalg.norm(fp_src - fp_tgt)

            if dist > merge_thresh:
                continue

            output_nodes.append(node_tgt)

    output_nodes = list(set(output_nodes))

    src = output_nodes
    nbrs = []
    for i in tqdm(range(n_hops-1)):
        for node in src:
            nbrs.extend(list(nx.all_neighbors(tgt_KG, node)))
        src = nbrs
        output_nodes.extend(nbrs)

    output_nodes = list(set(output_nodes))

    out_file = os.path.join(output_dir, f'outputs_{src_thresh}_{tgt_thresh}_{merge_thresh}_{e_ij_max_thresh}_{n_hops}.csv')

    # Write output to csv file
    print('Writing output to csv file...')
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for node in output_nodes:
            writer.writerow([node])