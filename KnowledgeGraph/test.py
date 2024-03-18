import glob
import os
import csv
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Compare')
parser.add_argument('--ml_out', type=str, required=True)
parser.add_argument('--kg_out_dir', type=str, required=True)
args = parser.parse_args()

# Get arguments
ml_out = args.ml_out
kg_out_dir = args.kg_out_dir

out_file = os.path.basename(ml_out)[:-4] + '_summary.csv'
out_file = open(out_file, 'w', newline='')

# out_file = open('summary.csv', 'w', newline='')

kg_files = glob.glob(os.path.join(kg_out_dir, '*.csv'))
# kg_files = glob.glob('KG_out_test/*.csv')

# Read ML output
ml = {}
ml_summary = {}
with open(ml_out, 'r') as f:
    print("Reading ML output...")
    reader = csv.reader(f)
    for row in tqdm(reader):
        if row[0] == 'ids':
            continue
        ml[row[0]] = float(row[1])

    add = 0
    count = 0
    count_greater_than_3 = 0
    count_less_than_3 = 0
    count_greater_than_2 = 0
    count_less_than_2 = 0
    count_greater_than_1 = 0
    count_less_than_1 = 0
    for key, val in ml.items():
        if val > 3:
            count_greater_than_3 += 1
        else:
            count_less_than_3 += 1

        if val > 2:
            count_greater_than_2 += 1
        else:
            count_less_than_2 += 1

        if val > 1:
            count_greater_than_1 += 1
        else:
            count_less_than_1 += 1
        add += val
        count += 1

    ml_summary['avg'] = add / count
    ml_summary['count'] = count
    ml_summary['count_greater_than_3'] = count_greater_than_3
    ml_summary['count_less_than_3'] = count_less_than_3
    ml_summary['count_greater_than_2'] = count_greater_than_2
    ml_summary['count_less_than_2'] = count_less_than_2
    ml_summary['count_greater_than_1'] = count_greater_than_1
    ml_summary['count_less_than_1'] = count_less_than_1

print("Reading KG outputs...")
for kg_file in tqdm(kg_files):

    kg_summary = {}

    props = os.path.basename(kg_file)[8:-4]
    # props - props.split('_')
    # src_thresh = props[0]
    # tgt_thresh = props[1]
    # merge_thresh = props[2]
    # e_ij_max_thresh = props[3]
    # n_hops = props[4]

    # output_dir = os.path.basename(kg_file)[:-4]
    # output_file = os.path.join(output_dir, f'{os.path.basename(ml_out).split(".")[0]}_summary.csv')

    kg_ids = []
    with open(kg_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            kg_ids.append(row[0])

    add = 0
    count = 0
    count_greater_than_3 = 0
    count_less_than_3 = 0
    count_greater_than_2 = 0
    count_less_than_2 = 0
    count_greater_than_1 = 0
    count_less_than_1 = 0
    for id in kg_ids:
        if id in ml:
            val = ml[id]
            if val > 3:
                count_greater_than_3 += 1
            else:
                count_less_than_3 += 1

            if val > 2:
                count_greater_than_2 += 1
            else:
                count_less_than_2 += 1

            if val > 1:
                count_greater_than_1 += 1
            else:
                count_less_than_1 += 1
            add += val
            count += 1

    if count == 0:
        count = 1
    avg = add / count
    filtered_greater_than_3 = count_greater_than_3 / ml_summary['count_greater_than_3'] * 100
    filtered_less_than_3 = count_less_than_3 / ml_summary['count_less_than_3'] * 100
    percent_greater_than_3 = count_greater_than_3 / count * 100
    percent_less_than_3 = count_less_than_3 / count * 100

    filtered_greater_than_2 = count_greater_than_2 / ml_summary['count_greater_than_2'] * 100
    filtered_less_than_2 = count_less_than_2 / ml_summary['count_less_than_2'] * 100
    percent_greater_than_2 = count_greater_than_2 / count * 100
    percent_less_than_2 = count_less_than_2 / count * 100

    filtered_greater_than_1 = count_greater_than_1 / ml_summary['count_greater_than_1'] * 100
    filtered_less_than_1 = count_less_than_1 / ml_summary['count_less_than_1'] * 100
    percent_greater_than_1 = count_greater_than_1 / count * 100
    percent_less_than_1 = count_less_than_1 / count * 100

    writer = csv.writer(out_file)
    # writer.writerow([f'src_thresh: {src_thresh}', f'tgt_thresh: {tgt_thresh}', f'merge_thresh: {merge_thresh}', f'e_ij_max_thresh: {e_ij_max_thresh}', f'n_hops: {n_hops}'])
    writer.writerow([f'{props}', f'{avg:.4f}', f'{count}', f'{count_greater_than_3}', f'{count_less_than_3}', f'{filtered_greater_than_3:.4f}', f'{filtered_less_than_3:.4f}', f'{percent_greater_than_3:.4f}', f'{percent_less_than_3:.4f}', f'{count_greater_than_2}', f'{count_less_than_2}', f'{filtered_greater_than_2:.4f}', f'{filtered_less_than_2:.4f}', f'{percent_greater_than_2:.4f}', f'{percent_less_than_2:.4f}', f'{count_greater_than_1}', f'{count_less_than_1}', f'{filtered_greater_than_1:.4f}', f'{filtered_less_than_1:.4f}', f'{percent_greater_than_1:.4f}', f'{percent_less_than_1:.4f}'])
    # writer.writerow([f'{props}',f'avg: {avg:.4f}', f'count: {count}', f'>3: {count_greater_than_3}', f'<3: {count_less_than_3}', f'filtered_greater_than_3: {filtered_greater_than_3:.4f}', f'filtered_less_than_3: {filtered_less_than_3:.4f}', f'percent_greater_than_3: {percent_greater_than_3:.4f}', f'percent_less_than_3: {percent_less_than_3:.4f}'])

out_file.close()