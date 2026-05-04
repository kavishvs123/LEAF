"""
Reads the choices dump produced by running with --selector_type optimal --dump, builds prompts matching the paper's Figure 8 template, queries a local LLaMA model via vllm, and writes the LLM selections to
data/llm_output/{dataset}_output.json.

Usage:
    python generate_llm_output.py --dataset PEMS08
    python generate_llm_output.py --dataset PEMS08 --model_path /path/to/llama --batch_size 64
"""

from vllm import LLM, SamplingParams
import argparse
import json
import os
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS08'])
parser.add_argument('--postfix', type=str, default='', help='Postfix used during the optimal selector dump (e.g. _noaug)')
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='HuggingFace model ID or local path to LLaMA weights')
parser.add_argument('--dump_dir', type=str, default='./outputs/dump')
parser.add_argument('--output_dir', type=str, default='./data/llm_output')
parser.add_argument('--batch_size', type=int, default=64, help='Number of prompts to send to vllm at once')
parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens — needs to be high enough for chain-of-thought reasoning')
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--round', type=int, default=1, choices=[1, 2],
                    help='Which round: 1 writes _output.json, 2 writes _output_r2.json')
args = parser.parse_args()

# ── Dataset metadata ──────────────────────────────────────────────────────────

DISTRICT = {'PEMS03': 3, 'PEMS04': 4, 'PEMS08': 8}[args.dataset]

# Human-readable augmentation labels matching the paper's Figure 8
AUG_LABELS = {
    'none':                   'none',
    'smoothed_output':        'smoothing the prediction of the model',
    'overestimated_output':   '5% higher than the prediction',
    'underestimated_output':  '5% lower than the prediction',
    'upward_trend_output':    'prediction with upward trend',
    'downward_trend_output':  'prediction with downward trend',
}

MODEL_LABELS = {
    'GraphBranch':      'the graph branch',
    'HypergraphBranch': 'the hypergraph branch',
}

# ── Load adjacency for spatial info ───────────────────────────────────────────
# Build a neighbour list from the edge CSV so we can describe each sensor's
# spatial context in the prompt, matching "<spatial information of the vertex>".

def load_neighbours(dataset: str) -> dict:
    """Return {node_idx: [neighbour_node_idx, ...]} from the dataset CSV."""
    csv_path = f'./data/{dataset}/{dataset}.csv'
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    neighbours = {}
    for _, row in df.iterrows():
        src, dst = int(row.iloc[0]), int(row.iloc[1])
        neighbours.setdefault(src, []).append(dst)
        neighbours.setdefault(dst, []).append(src)
    return neighbours

neighbours = load_neighbours(args.dataset)

# ── File paths ─────────────────────────────────────────────────────────────────

dataset = args.dataset.lower()
postfix = args.postfix

if args.round == 1:
    choices_path = os.path.join(args.dump_dir, f'{dataset}_choices{postfix}.json')
    output_path  = os.path.join(args.output_dir, f'{dataset}_output{postfix}.json')
else:
    choices_path = os.path.join(args.dump_dir, f'{dataset}_choices_r2{postfix}.json')
    output_path  = os.path.join(args.output_dir, f'{dataset}_output_r2{postfix}.json')

os.makedirs(args.output_dir, exist_ok=True)

print(f'Loading choices from: {choices_path}')
with open(choices_path, 'r') as f:
    entries = json.load(f)

if len(entries) == 0:
    raise ValueError(
        f'{choices_path} is empty. '
        f'Run the optimal selector first:\n'
        f'  python main.py --dataset {args.dataset} --method select '
        f'--model_list GraphBranch HypergraphBranch '
        f'--adapter_type aug --selector_type optimal '
        f'--update_iters 0 '
        f'--ckpt_paths <graph.pth> <hypergraph.pth> --dump'
    )

print(f'Loaded {len(entries)} entries.')

# ── Prompt builder (matching paper Figure 8) ───────────────────────────────────

def build_prompt(entry: dict) -> str:
    node_idx       = entry['node_idx']
    history        = entry['history']        # list of ints, length seq_in_len
    choices        = entry['choices']        # list of lists, each length seq_out_len
    model_names    = entry['model_names']
    aug_types      = entry['aug_types']
    input_start    = entry['input_start_time']
    input_end      = entry['input_end_time']
    output_start   = entry['output_start_time']
    output_end     = entry['output_end_time']

    # Spatial info: list the directly connected sensor IDs
    neighbour_ids = neighbours.get(node_idx, [])
    if neighbour_ids:
        spatial_info = f'connected to sensors {neighbour_ids}'
    else:
        spatial_info = 'spatial information unavailable'

    history_str  = ', '.join(str(v) for v in history)
    history_avg  = round(sum(history) / len(history), 1) if history else 0

    # Build numbered candidate list matching the paper's format
    candidates_str = ''
    for idx, (pred, model, aug) in enumerate(zip(choices, model_names, aug_types), start=1):
        pred_str  = ', '.join(str(v) for v in pred)
        pred_avg  = round(sum(pred) / len(pred), 1)
        aug_label = AUG_LABELS.get(aug, aug)
        mod_label = MODEL_LABELS.get(model, model)
        candidates_str += (
            f'{idx}.\t[{pred_str}] (average: {pred_avg}), '
            f'{mod_label}, Augmentation: {aug_label}\n'
        )

    prompt = (
        f'Given historical data for traffic flow over 12 time steps at sensor {node_idx} '
        f'of District {DISTRICT}, California ({spatial_info}), '
        f'the recorded traffic flows are [{history_str}] (average: {history_avg}), '
        f'from {input_start} to {input_end}, '
        f'with data points recorded at 5-minute intervals. '
        f'We aim to predict the traffic flow in the next 12 time steps '
        f'from {output_start} to {output_end}. '
        f'We use two branches: the graph branch (which captures pair-wise relations), '
        f'the hypergraph branch (which captures non-pair-wise relations) to predict '
        f'the future traffic flow and perform some augmentations (transformations) to '
        f'the predictions. Note that smoothing does not reduce MAE error. '
        f'Here are some choices for you. Please consider: (1) historical values, '
        f'(2) models\' predictions, and (3) the time of the data (most importantly, '
        f'whether they are in beginning / end of the morning / afternoon rush hours. '
        f'Note that different nodes have different rush hours, if you have seen '
        f'significant changes in history, you can identify the beginning / end of '
        f'the rush hour).\n'
        f'Please first think carefully and then select the most likely one:\n'
        f'{candidates_str}'
    )
    return prompt


# ── Parse LLM response ─────────────────────────────────────────────────────────

def parse_answer(text: str, num_candidates: int):
    """
    Return 1-indexed int, or None if unparseable.

    The paper's prompt says 'think carefully then select', so the LLM reasons
    before giving its answer. We take the LAST valid integer in [1, num_candidates]
    in the response rather than the first.
    """
    matches = re.findall(r'\b(\d+)\b', text)
    for m in reversed(matches):
        val = int(m)
        if 1 <= val <= num_candidates:
            return val
    return None


# ── vllm inference ─────────────────────────────────────────────────────────────

print(f'Loading model: {args.model_path}')

llm = LLM(model=args.model_path)
sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_tokens,
)

# Build all prompts upfront
print('Building prompts...')
prompts = [build_prompt(e) for e in entries]

# Run inference in batches
print(f'Running inference on {len(prompts)} prompts (batch size {args.batch_size})...')
raw_outputs = []
for start in range(0, len(prompts), args.batch_size):
    batch = prompts[start:start + args.batch_size]
    results = llm.generate(batch, sampling_params)
    raw_outputs.extend(results)
    if (start // args.batch_size) % 10 == 0:
        print(f'  Processed {min(start + args.batch_size, len(prompts))} / {len(prompts)}')

# ── Parse and save ─────────────────────────────────────────────────────────────

output_data = []
none_count  = 0

for entry, result in zip(entries, raw_outputs):
    text = result.outputs[0].text
    num_candidates = len(entry['choices'])
    answer = parse_answer(text, num_candidates)
    if answer is None:
        none_count += 1
    output_data.append({'final_answer': answer, 'raw_response': text})

print(
    f'Done. Unparseable responses: {none_count} / {len(entries)} '
    f'(these default to candidate 1 at inference time)')

with open(output_path, 'w') as f:
    json.dump(output_data, f)

print(f'Saved to: {output_path}')
