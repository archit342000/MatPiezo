import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Compare')
parser.add_argument('--ml_out', type=str, required=True)
parser.add_argument('--kg_out', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='output', help='output directory')
args = parser.parse_args()

# Get arguments
kg_out = args.kg_out
ml_out = args.ml_out
output_dir = args.output_dir

# Check if output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read KG output
kg = []
with open(kg_out, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        kg.append(row[0])

# Read ML output
ml = {}
with open(ml_out, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'ids':
            continue
        ml[row[0]] = float(row[1])

out_file = open(os.path.join(output_dir, f'{os.path.basename(ml_out)[:-4]}_filtered.csv'), 'w', newline='')
writer = csv.writer(out_file)

# Get values of keys in kg_out
for id in kg:
    if id in ml:
        val = ml[id]
        writer.writerow([id, val])

out_file.close()