import os
import glob
from multiprocessing import Process
from tqdm import tqdm
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core.structure import Structure
import warnings
import numpy as np
import glob
import argparse
import time

warnings.filterwarnings("ignore")

def get_fingerprints(out_dir, files, count):
    fingerprint_path = os.path.join(out_dir, f'fingerprints_{count}.npy')

    ssf = SiteStatsFingerprint(CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0.0), stats=('mean', 'std_dev', 'minimum', 'maximum'))

    fps = {}
    for file in tqdm(files):
        try:
            id = os.path.basename(file).split('.')[0]
            st = Structure.from_file(file)
            fp = np.array(ssf.featurize(st))
            fps[id] = fp
        except:
            pass

    np.save(fingerprint_path, fps)

if __name__ == '__main__':

    print('Start')
    t1 = time.time()

    parser = argparse.ArgumentParser(description='Get fingerprints from structures')
    parser.add_argument('--dir', default='cifs', help='Directory containing CIF files')
    parser.add_argument('--out_dir', default='fingerprints', help='Directory to save fingerprints')
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of files to process in parallel')
    args = parser.parse_args()

    dir = args.dir
    chunk_size = args.chunk_size
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = glob.glob(f'{dir}/*.cif')

    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    processes = []

    for i in range(len(chunks)):
        p = Process(target=get_fingerprints, args=(out_dir, chunks[i], i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    t2 = time.time()

    print(f'Time taken: {t2 - t1:.2f} s')
    print('Done')