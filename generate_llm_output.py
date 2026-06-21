"""
generate_llm_output.py

Reads the choices dump produced by running with --selector_type optimal --dump,
builds prompts matching the paper's Figure 8 template, queries a local LLaMA
model via vllm, and writes the LLM selections to a per-GPU JSONL file.

Uses streaming JSON input (ijson) and line-by-line output (JSONL) so that
RAM usage stays constant regardless of dataset size. Supports resume if killed.

Once all three GPU scripts finish, run combine_llm_output.py to merge the
three JSONL files into the final pems08_output.json.

Requirements:
    pip install ijson

Usage — run each in a separate terminal simultaneously:
    CUDA_VISIBLE_DEVICES=0 python generate_llm_output.py --dataset PEMS08 --model_path meta-llama/Llama-3.1-8B-Instruct --start_idx 0 --end_idx 808180
    CUDA_VISIBLE_DEVICES=1 python generate_llm_output.py --dataset PEMS08 --model_path meta-llama/Llama-3.1-8B-Instruct --start_idx 808180 --end_idx 1616360
    CUDA_VISIBLE_DEVICES=2 python generate_llm_output.py --dataset PEMS08 --model_path meta-llama/Llama-3.1-8B-Instruct --start_idx 1616360 --end_idx 2424540
"""

import argparse
import json
import os
import re
import pandas as pd
import ijson

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS08'])
parser.add_argument('--postfix', type=str, default='', help='Postfix of the input choices file from the optimal selector dump (e.g. _noaug)')
parser.add_argument('--output_postfix', type=str, default=None, help='Postfix for the output file (e.g. _lora, _qlora). Defaults to --postfix if not set.')
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='HuggingFace model ID or local path to LLaMA weights')
parser.add_argument('--dump_dir', type=str, default='./outputs/dump')
parser.add_argument('--output_dir', type=str, default='./data/llm_output')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of prompts per vllm call')
parser.add_argument('--chunk_size', type=int, default=1000,
                    help='Number of entries to process before flushing to disk')
parser.add_argument('--max_tokens', type=int, default=128,
                    help='Max output tokens per prompt (chain-of-thought)')
# [ADDED] Separate token budget for fine-tuned LoRA models — they output just
# a single digit then stop, so 128 causes degenerate repetition loops.
parser.add_argument('--max_tokens_lora', type=int, default=5,
                    help='Max output tokens when --lora_path is set. Fine-tuned models '
                         'output a single digit so 5 is sufficient and prevents repetition.')
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--max_model_len', type=int, default=16384,
                    help='Max sequence length for vllm KV cache — reduce if GPU OOM')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.95,
                    help='Fraction of GPU memory vllm can use for KV cache')
parser.add_argument('--enforce_eager', action='store_true', default=True,
                    help='Disable CUDA graphs to free 1-3GB extra GPU memory')
parser.add_argument('--round', type=int, default=1, choices=[1, 2],
                    help='Which round: 1 writes _output.json, 2 writes _output_r2.json')
parser.add_argument('--lora_path', type=str, default=None,
                    help='Path to a fine-tuned LoRA/QLoRA adapter directory. '
                         'If provided, the adapter is loaded on top of --model_path.')
parser.add_argument('--lora_r', type=int, default=16,
                    help='LoRA rank used during fine-tuning (must match adapter)')
parser.add_argument('--prime', action='store_true', default=False,
                    help='Append the response template to each prompt so the model '
                         'completes with a number rather than continuing the candidate list')
parser.add_argument('--save_prompts', type=int, default=0,
                    help='Save the first N examples (prompt + response + ground truth) '
                         'to a JSON file for analysis/evaluation. 0 = disabled.')
parser.add_argument('--start_idx', type=int, default=0,
                    help='Global entry index to start from (inclusive)')
parser.add_argument('--end_idx', type=int, default=None,
                    help='Global entry index to stop at (exclusive). Defaults to end of file.')
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

dataset        = args.dataset.lower()
postfix        = args.postfix                                          # controls input choices file
output_postfix = args.output_postfix if args.output_postfix is not None else args.postfix  # controls output file
end_idx        = args.end_idx  # may be None (process to end of file)

if args.round == 1:
    choices_path = os.path.join(args.dump_dir, f'{dataset}_choices{postfix}.json')
    base_name    = f'{dataset}_output{output_postfix}'
else:
    choices_path = os.path.join(args.dump_dir, f'{dataset}_choices_r2{postfix}.json')
    base_name    = f'{dataset}_output_r2{output_postfix}'

# Each GPU writes to its own JSONL file named with its index range
jsonl_filename = f'{base_name}_{args.start_idx}_{end_idx}.jsonl'
jsonl_path     = os.path.join(args.output_dir, jsonl_filename)

os.makedirs(args.output_dir, exist_ok=True)

print(f'GPU range: entries {args.start_idx} to {end_idx}')
print(f'Output file: {jsonl_path}')


# ── Resume: count already-completed lines in this GPU's JSONL ─────────────────

completed_count = 0
none_count      = 0

if os.path.exists(jsonl_path):
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    completed_count = len(lines)
    none_count      = sum(1 for l in lines if json.loads(l)['final_answer'] is None)
    resume_from     = args.start_idx + completed_count
    print(f'Resuming from global entry {resume_from} ({completed_count} already done).')
else:
    resume_from = args.start_idx
    print('Starting fresh run.')


# ── Prompt builder (matching paper Figure 8) ──────────────────────────────────

def build_prompt(entry: dict) -> str:
    node_idx     = entry['node_idx']
    history      = entry['history']
    choices      = entry['choices']
    model_names  = entry['model_names']
    aug_types    = entry['aug_types']
    input_start  = entry['input_start_time']
    input_end    = entry['input_end_time']
    output_start = entry['output_start_time']
    output_end   = entry['output_end_time']

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
        + ('\n### Selected candidate: ' if args.prime else '')
    )


# ── Parse LLM response ─────────────────────────────────────────────────────────

def parse_answer(text: str, num_candidates: int, first: bool = False):
    """Extract a valid integer in [1, num_candidates] from the model's response.

    first=True  — take the first valid integer (use when --prime prepends the
                  response template; the model outputs the answer before reasoning)
    first=False — take the last valid integer (use without --prime; the model
                  reasons first and gives the answer at the end)
    """
    matches = re.findall(r'\b(\d+)\b', text)
    if first:
        for m in matches:
            val = int(m)
            if 1 <= val <= num_candidates:
                return val
    else:
        for m in reversed(matches):
            val = int(m)
            if 1 <= val <= num_candidates:
                return val
    return None


# ── vllm inference ─────────────────────────────────────────────────────────────

print(f'Loading model: {args.model_path}')
if args.lora_path:
    print(f'Using LoRA adapter: {args.lora_path}')
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

if args.lora_path:
    llm = LLM(
        model=args.model_path,
        enable_lora=True,
        max_lora_rank=args.lora_r,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    lora_request = LoRARequest('finetuned', 1, args.lora_path)
else:
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    lora_request = None

# [CHANGED] When using a fine-tuned LoRA adapter:
#   - cap output at max_tokens_lora (default 5) — the model outputs one digit then stops,
#     so 128 tokens causes degenerate repetition loops filling the token budget
#   - stop on token ID 128009 (<|eot_id|> in LLaMA 3.1) so vllm halts where the model
#     was trained to stop. stop_token_ids is used instead of stop=["<|eot_id|>"] because
#     vllm matches stop strings against decoded text, which may not reliably trigger on
#     special tokens — stop_token_ids matches at the token level before decoding.
if args.lora_path:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens_lora,
        stop_token_ids=[128009],
    )
else:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

tokenizer         = AutoTokenizer.from_pretrained(args.model_path)
max_prompt_tokens = args.max_model_len - (args.max_tokens_lora if args.lora_path else args.max_tokens)

def truncate_prompt(prompt: str) -> str:
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_prompt_tokens:
        tokens = tokens[:max_prompt_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return prompt


# ── Main loop — stream input, write output line by line ───────────────────────

chunk_entries     = []
global_idx        = 0
saved_examples    = []   # accumulated for --save_prompts
examples_written  = False  # [ADDED] flag so we only write the examples file once

# [ADDED] Write examples to disk as soon as all save_prompts examples are collected.
# Previously this only happened at the very end of the run (hours later). Now the file
# appears after the first ~save_prompts entries are processed.
def _flush_examples():
    global examples_written
    if examples_written or not saved_examples:
        return
    n_correct     = sum(1 for e in saved_examples if e['correct'])
    accuracy      = n_correct / len(saved_examples) * 100
    examples_path = os.path.join(
        args.output_dir,
        f'{base_name}_{args.start_idx}_{end_idx}_examples.json'
    )
    with open(examples_path, 'w') as f:
        json.dump({'accuracy': accuracy, 'examples': saved_examples}, f, indent=2)
    examples_written = True
    print(f'\nSaved {len(saved_examples)} examples to: {examples_path}')
    print(f'Accuracy on saved examples: {accuracy:.1f}% ({n_correct}/{len(saved_examples)})')

def flush_chunk(chunk_entries, jsonl_file):
    """Run vllm on a chunk and append results to the JSONL file."""
    global none_count, saved_examples
    # [CHANGED] When using a fine-tuned LoRA adapter, wrap prompts in the LLaMA chat
    # template to match the format used during fine-tuning. Without these special tokens
    # the fine-tuned model won't recognise the input format correctly.
    if args.lora_path:
        prompts = [
            '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
            + truncate_prompt(build_prompt(e))
            + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            for e in chunk_entries
        ]
    else:
        prompts = [truncate_prompt(build_prompt(e)) for e in chunk_entries]
    results = []
    for batch_start in range(0, len(prompts), args.batch_size):
        batch   = prompts[batch_start:batch_start + args.batch_size]
        outputs = llm.generate(batch, sampling_params, lora_request=lora_request)
        results.extend(outputs)
    for entry, prompt, result in zip(chunk_entries, prompts, results):
        text           = result.outputs[0].text
        num_candidates = len(entry['choices'])
        # [CHANGED] Fine-tuned model outputs the answer first (trained to do so);
        # use first=True in that case regardless of --prime flag.
        answer         = parse_answer(text, num_candidates, first=args.prime or bool(args.lora_path))
        if answer is None:
            none_count += 1
        jsonl_file.write(json.dumps({'final_answer': answer, 'raw_response': text}) + '\n')

        # Save examples for analysis if requested
        if args.save_prompts > 0 and len(saved_examples) < args.save_prompts:
            saved_examples.append({
                'node_idx':          entry['node_idx'],
                'input_start_time':  entry['input_start_time'],
                'input_end_time':    entry['input_end_time'],
                'output_start_time': entry['output_start_time'],
                'output_end_time':   entry['output_end_time'],
                'optimal_idx':       entry['optimal_idx'],          # ground truth (0-indexed)
                'optimal_answer':    entry['optimal_idx'] + 1,      # ground truth (1-indexed)
                'llm_answer':        answer,                        # LLM selection (1-indexed)
                'correct':           answer == entry['optimal_idx'] + 1,
                'prompt':            prompt,
                'raw_response':      text,
            })
            # [ADDED] Write to disk as soon as we have collected all requested examples
            # so the file is available immediately rather than at end of the full run.
            if len(saved_examples) == args.save_prompts:
                _flush_examples()

    jsonl_file.flush()

print(f'Streaming entries from {choices_path}...')

with open(jsonl_path, 'a') as jsonl_file:
    with open(choices_path, 'rb') as f:
        for entry in ijson.items(f, 'item'):

            # Skip entries before this GPU's range
            if global_idx < args.start_idx:
                global_idx += 1
                continue

            # Stop at end of this GPU's range
            if end_idx is not None and global_idx >= end_idx:
                break

            # Skip entries already completed in a previous run
            if global_idx < resume_from:
                global_idx += 1
                continue

            chunk_entries.append(entry)
            global_idx += 1

            if len(chunk_entries) == args.chunk_size:
                flush_chunk(chunk_entries, jsonl_file)
                chunk_entries = []
                print(f'  Saved progress: {global_idx} / {end_idx} '
                      f'(unparseable so far: {none_count})')

    # Final partial chunk
    if chunk_entries:
        flush_chunk(chunk_entries, jsonl_file)
        print(f'  Saved progress: {global_idx} / {end_idx} '
              f'(unparseable so far: {none_count})')

print(f'\nGPU chunk complete. Entries {args.start_idx} to {end_idx} done.')
print(f'Unparseable: {none_count}')
print(f'Output: {jsonl_path}')
print(f'Run combine_llm_output.py once all GPUs are finished.')

# ── Save prompt examples for analysis ────────────────────────────────────────

# [CHANGED] Use _flush_examples() which is a no-op if already written mid-run.
# This handles the edge case where the total entries < save_prompts.
if args.save_prompts > 0:
    _flush_examples()
