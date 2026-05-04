"""
generate_llm_output.py

Reads the choices dump produced by running with --selector_type optimal --dump,
builds prompts matching the paper's Figure 8 template, queries a local LLaMA
model via vllm, and writes the LLM selections to
data/llm_output/{dataset}_output.json.

Processes entries in chunks to keep RAM usage low, and saves progress
incrementally so the run can be resumed if interrupted.

Usage:
    python generate_llm_output.py --dataset PEMS08
    python generate_llm_output.py --dataset PEMS08 --model_path /path/to/llama
"""

import argparse
import json
import os
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS08'])
parser.add_argument('--postfix', type=str, default='', help='Postfix used during the optimal selector dump (e.g. _noaug)')
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='HuggingFace model ID or local path to LLaMA weights')
parser.add_argument('--dump_dir', type=str, default='./outputs/dump')
parser.add_argument('--output_dir', type=str, default='./data/llm_output')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of prompts per vllm call')
parser.add_argument('--chunk_size', type=int, default=1000,
                    help='Number of entries to process before saving progress to disk')
parser.add_argument('--max_tokens', type=int, default=128,
                    help='Max output tokens per prompt (chain-of-thought)')
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--max_model_len', type=int, default=16384,
                    help='Max sequence length for vllm KV cache — reduce if GPU OOM')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.95,
                    help='Fraction of GPU memory vllm can use for KV cache')
parser.add_argument('--enforce_eager', action='store_true', default=True,
                    help='Disable CUDA graphs to free 1-3GB extra GPU memory')
parser.add_argument('--round', type=int, default=1, choices=[1, 2],
                    help='Which round: 1 writes _output.json, 2 writes _output_r2.json')
args = parser.parse_args()


# ── Dataset metadata ──────────────────────────────────────────────────────────

DISTRICT = {'PEMS03': 3, 'PEMS04': 4, 'PEMS08': 8}[args.dataset]

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

def load_neighbours(dataset: str) -> dict:
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

# Temporary file written to incrementally — renamed to output_path on completion
tmp_output_path = output_path + '.tmp'

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

total = len(entries)
print(f'Loaded {total} entries.')


# ── Resume from previous partial run ─────────────────────────────────────────

start_idx = 0
completed = []

if os.path.exists(tmp_output_path):
    with open(tmp_output_path, 'r') as f:
        completed = json.load(f)
    start_idx = len(completed)
    print(f'Resuming from entry {start_idx} / {total} ({start_idx} already done).')
else:
    print('Starting fresh run.')


# ── Prompt builder (matching paper Figure 8) ──────────────────────────────────

def build_prompt(entry: dict) -> str:
    node_idx    = entry['node_idx']
    history     = entry['history']
    choices     = entry['choices']
    model_names = entry['model_names']
    aug_types   = entry['aug_types']
    input_start = entry['input_start_time']
    input_end   = entry['input_end_time']
    output_start = entry['output_start_time']
    output_end  = entry['output_end_time']

    neighbour_ids = neighbours.get(node_idx, [])
    spatial_info  = f'connected to sensors {neighbour_ids}' if neighbour_ids else 'spatial information unavailable'

    history_str = ', '.join(str(v) for v in history)
    history_avg = round(sum(history) / len(history), 1) if history else 0

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

    return (
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


# ── Parse LLM response ─────────────────────────────────────────────────────────

def parse_answer(text: str, num_candidates: int):
    """Take the last valid integer in [1, num_candidates] — LLM reasons before answering."""
    for m in reversed(re.findall(r'\b(\d+)\b', text)):
        val = int(m)
        if 1 <= val <= num_candidates:
            return val
    return None


# ── vllm inference ─────────────────────────────────────────────────────────────

print(f'Loading model: {args.model_path}')
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

llm = LLM(
    model=args.model_path,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    enforce_eager=args.enforce_eager,
)
sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_tokens,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
max_prompt_tokens = args.max_model_len - args.max_tokens

def truncate_prompt(prompt: str) -> str:
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_prompt_tokens:
        tokens = tokens[:max_prompt_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return prompt


# ── Main loop — process in chunks, save after each chunk ─────────────────────

none_count = sum(1 for r in completed if r['final_answer'] is None)

print(f'Processing {total - start_idx} remaining entries '
      f'(chunk size {args.chunk_size}, batch size {args.batch_size})...')

for chunk_start in range(start_idx, total, args.chunk_size):
    chunk_end     = min(chunk_start + args.chunk_size, total)
    chunk_entries = entries[chunk_start:chunk_end]

    # Build and truncate prompts for this chunk only
    chunk_prompts = [truncate_prompt(build_prompt(e)) for e in chunk_entries]

    # Run vllm in batches within the chunk
    chunk_results = []
    for batch_start in range(0, len(chunk_prompts), args.batch_size):
        batch   = chunk_prompts[batch_start:batch_start + args.batch_size]
        outputs = llm.generate(batch, sampling_params)
        chunk_results.extend(outputs)

    # Parse results and append to completed list
    for entry, result in zip(chunk_entries, chunk_results):
        text           = result.outputs[0].text
        num_candidates = len(entry['choices'])
        answer         = parse_answer(text, num_candidates)
        if answer is None:
            none_count += 1
        completed.append({'final_answer': answer, 'raw_response': text})

    # Save progress to tmp file after every chunk
    with open(tmp_output_path, 'w') as f:
        json.dump(completed, f)

    print(f'  Saved progress: {chunk_end} / {total} '
          f'(unparseable so far: {none_count})')

# ── Finalise ──────────────────────────────────────────────────────────────────

os.replace(tmp_output_path, output_path)
print(f'\nDone. Unparseable responses: {none_count} / {total} '
      f'(these default to candidate 1 at inference time)')
print(f'Saved to: {output_path}')
