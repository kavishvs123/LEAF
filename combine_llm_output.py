"""
combine_llm_output.py

Combines the three per-GPU JSONL files produced by generate_llm_output.py
into the final JSON file required by LLMSelectorFromJson.

Run this once all three GPU scripts have finished.

Usage:
    python combine_llm_output.py --dataset PEMS08
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS08'])
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--output_dir', type=str, default='./data/llm_output')
parser.add_argument('--round', type=int, default=1, choices=[1, 2])
# The three index ranges — must match what was used in generate_llm_output.py
parser.add_argument('--splits', type=int, nargs='+', default=[0, 808180, 1616360, 2424540],
                    help='Boundary indices for the three GPU splits e.g. 0 808180 1616360 2424540')
args = parser.parse_args()

dataset = args.dataset.lower()
postfix = args.postfix

if args.round == 1:
    base_name   = f'{dataset}_output{postfix}'
else:
    base_name   = f'{dataset}_output_r2{postfix}'

output_path = os.path.join(args.output_dir, f'{base_name}.json')

# Build the three expected JSONL filenames in order
splits      = args.splits
jsonl_files = []
for i in range(len(splits) - 1):
    start = splits[i]
    end   = splits[i + 1]
    fname = f'{base_name}_{start}_{end}.jsonl'
    fpath = os.path.join(args.output_dir, fname)
    jsonl_files.append((start, end, fpath))

# Verify all files exist and are complete before combining
print('Checking JSONL files...')
all_ok = True
for start, end, fpath in jsonl_files:
    if not os.path.exists(fpath):
        print(f'  MISSING: {fpath}')
        all_ok = False
    else:
        with open(fpath, 'r') as f:
            count = sum(1 for _ in f)
        expected = end - start
        status   = 'OK' if count == expected else f'INCOMPLETE — expected {expected}, got {count}'
        print(f'  {fpath}: {count} entries — {status}')
        if count != expected:
            all_ok = False

if not all_ok:
    print('\nOne or more files are missing or incomplete. '
          'Wait for all GPU scripts to finish before combining.')
    exit(1)

# Combine in order and write final JSON
print('\nCombining files...')
combined = []
for start, end, fpath in jsonl_files:
    with open(fpath, 'r') as f:
        for line in f:
            combined.append(json.loads(line))
    print(f'  Added {end - start} entries from {fpath}')

print(f'\nTotal entries: {len(combined)}')

with open(output_path, 'w') as f:
    json.dump(combined, f)

print(f'Saved to: {output_path}')

# Clean up JSONL files
for _, _, fpath in jsonl_files:
    os.remove(fpath)
    print(f'Removed: {fpath}')

print('\nDone. Ready to run LLM selector fine-tuning.')
