import os
import time
import argparse

parser = argparse.ArgumentParser(description='Create KG')
parser.add_argument('--src_fps', type=str, default = 'merged_piezo/full_fingerprints.npy', help='source KG file')
parser.add_argument('--tgt_fps', type=str, default = 'COD/full_fingerprints.npy', help='target KG file')
parser.add_argument('--output_dir', type=str, default='KG_out', help='Path to output directory')
args = parser.parse_args()

# Get arguments
src_fps = args.src_fps
tgt_fps = args.tgt_fps
output_dir = args.output_dir

tgt_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
merge_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
eij_max_threshs = [1.0, 2.0, 3.0]
n_hops = [1, 2]

for tgt_thresh in tgt_threshs:
    for merge_thresh in merge_threshs:
        for eij_max_thresh in eij_max_threshs:
            for n_hop in n_hops:
                t1 = time.time()
                os.system(f'python filter_from_KG.py --src_fps {src_fps} --tgt_fps {tgt_fps} --src_thresh 0.9 --tgt_thresh {tgt_thresh} --merge_thresh {merge_thresh} --e_ij_max_thresh {eij_max_thresh} --n_hops {n_hop} --output_dir {output_dir}')
                t2 = time.time()
                print(f'Completed KG_{tgt_thresh}_{merge_thresh}_{eij_max_thresh}_{n_hop}')
                print(f'Time taken: {t2-t1} seconds')