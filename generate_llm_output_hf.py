"""
generate_llm_output_hf.py

HuggingFace Transformers version of generate_llm_output.py.
Designed specifically for QLoRA adapters, which require the base model
to be loaded in 4-bit (matching training conditions). vllm applies LoRA
adapters on top of a full-precision base model, which causes a mismatch
with QLoRA adapters trained on a 4-bit base — this script avoids that.

Slower than vllm (~10-20x) but correctly matches QLoRA training conditions.
Output format is identical to generate_llm_output.py, so combine_llm_output.py
works unchanged.

Requirements:
    pip install transformers peft bitsandbytes ijson

Usage — run each in a separate terminal simultaneously:
    CUDA_VISIBLE_DEVICES=0 python generate_llm_output_hf.py --dataset PEMS08 --lora_path ./outputs/qlora_full_training/pems08_qlora_adapter --lora_r 64 --output_postfix _qlora_full_train --save_prompts 500 --start_idx 0 --end_idx 808180
    CUDA_VISIBLE_DEVICES=1 python generate_llm_output_hf.py --dataset PEMS08 --lora_path ./outputs/qlora_full_training/pems08_qlora_adapter --lora_r 64 --output_postfix _qlora_full_train --start_idx 808180 --end_idx 1616360
    CUDA_VISIBLE_DEVICES=2 python generate_llm_output_hf.py --dataset PEMS08 --lora_path ./outputs/qlora_full_training/pems08_qlora_adapter --lora_r 64 --output_postfix _qlora_full_train --start_idx 1616360 --end_idx 2424540
"""

import argparse
import json
import os
import re
import torch
import pandas as pd
import ijson

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS08'])
parser.add_argument('--postfix', type=str, default='', help='Postfix of the input choices file (e.g. _noaug)')
parser.add_argument('--output_postfix', type=str, default=None, help='Postfix for the output file. Defaults to --postfix if not set.')
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='HuggingFace model ID or local path to LLaMA weights')
parser.add_argument('--lora_path', type=str, default=None,
                    help='Path to a fine-tuned QLoRA adapter directory.')
parser.add_argument('--lora_r', type=int, default=64,
                    help='LoRA rank used during fine-tuning (must match adapter)')
parser.add_argument('--dump_dir', type=str, default='./outputs/dump')
parser.add_argument('--output_dir', type=str, default='./data/llm_output')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Number of prompts per model.generate() call. '
                         'Lower than vllm default due to KV cache memory with long prompts.')
parser.add_argument('--chunk_size', type=int, default=1000,
                    help='Number of entries to process before flushing to disk')
parser.add_argument('--max_new_tokens', type=int, default=5,
                    help='Max new tokens to generate. Fine-tuned model outputs a single '
                         'digit so 5 is sufficient.')
parser.add_argument('--max_prompt_tokens', type=int, default=2048,
                    help='Truncate prompts longer than this many tokens')
parser.add_argument('--round', type=int, default=1, choices=[1, 2],
                    help='Which round: 1 writes _output.json, 2 writes _output_r2.json')
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

EOT_TOKEN_ID = 128009  # <|eot_id|> in LLaMA 3.1


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


# ── File paths ────────────────────────────────────────────────────────────────

dataset        = args.dataset.lower()
postfix        = args.postfix
output_postfix = args.output_postfix if args.output_postfix is not None else args.postfix
end_idx        = args.end_idx

if args.round == 1:
    choices_path = os.path.join(args.dump_dir, f'{dataset}_choices{postfix}.json')
    base_name    = f'{dataset}_output{output_postfix}'
else:
    choices_path = os.path.join(args.dump_dir, f'{dataset}_choices_r2{postfix}.json')
    base_name    = f'{dataset}_output_r2{output_postfix}'

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
    )


# ── Parse LLM response ────────────────────────────────────────────────────────

def parse_answer(text: str, num_candidates: int):
    """Extract the first valid integer in [1, num_candidates] from the model response.
    Fine-tuned model outputs the answer first before any reasoning, so we take first match.
    """
    matches = re.findall(r'\b(\d+)\b', text)
    for m in matches:
        val = int(m)
        if 1 <= val <= num_candidates:
            return val
    return None


# ── Load model in 4-bit (matching QLoRA training conditions) ──────────────────

print(f'Loading model: {args.model_path}')
if args.lora_path:
    print(f'Using QLoRA adapter: {args.lora_path}')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
)

if args.lora_path:
    model = PeftModel.from_pretrained(model, args.lora_path)

model.eval()

# Left-pad inputs so all sequences in a batch end at the same position,
# allowing model.generate() to produce new tokens correctly for all items.
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token    = tokenizer.eos_token

print('Model loaded.')


# ── Main loop — stream input, write output line by line ───────────────────────

chunk_entries    = []
global_idx       = 0
saved_examples   = []
examples_written = False


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
    """Run HuggingFace inference on a chunk and append results to the JSONL file."""
    global none_count, saved_examples

    # Wrap each prompt in the LLaMA 3.1 chat template to match training format
    prompts = [
        '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
        + build_prompt(e)
        + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        for e in chunk_entries
    ]

    # Truncate prompts that exceed max_prompt_tokens
    truncated_prompts = []
    for p in prompts:
        tokens = tokenizer.encode(p)
        if len(tokens) > args.max_prompt_tokens:
            tokens = tokens[:args.max_prompt_tokens]
            p = tokenizer.decode(tokens, skip_special_tokens=False)
        truncated_prompts.append(p)

    all_texts = []
    for batch_start in range(0, len(truncated_prompts), args.batch_size):
        batch_prompts = truncated_prompts[batch_start:batch_start + args.batch_size]

        encodings = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
        ).to(device)

        input_len = encodings['input_ids'].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=EOT_TOKEN_ID,
            )

        # Decode only the newly generated tokens (not the input prompt)
        new_token_ids = output_ids[:, input_len:]
        texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        all_texts.extend(texts)

    # Free GPU memory accumulated during generation
    torch.cuda.empty_cache()

    for entry, prompt, text in zip(chunk_entries, truncated_prompts, all_texts):
        num_candidates = len(entry['choices'])
        answer         = parse_answer(text, num_candidates)
        if answer is None:
            none_count += 1
        jsonl_file.write(json.dumps({'final_answer': answer, 'raw_response': text}) + '\n')

        if args.save_prompts > 0 and len(saved_examples) < args.save_prompts:
            saved_examples.append({
                'node_idx':          entry['node_idx'],
                'input_start_time':  entry['input_start_time'],
                'input_end_time':    entry['input_end_time'],
                'output_start_time': entry['output_start_time'],
                'output_end_time':   entry['output_end_time'],
                'optimal_idx':       entry['optimal_idx'],
                'optimal_answer':    entry['optimal_idx'] + 1,
                'llm_answer':        answer,
                'correct':           answer == entry['optimal_idx'] + 1,
                'prompt':            prompt,
                'raw_response':      text,
            })
            if len(saved_examples) == args.save_prompts:
                _flush_examples()

    jsonl_file.flush()


print(f'Streaming entries from {choices_path}...')

with open(jsonl_path, 'a') as jsonl_file:
    with open(choices_path, 'rb') as f:
        for entry in ijson.items(f, 'item'):

            if global_idx < args.start_idx:
                global_idx += 1
                continue

            if end_idx is not None and global_idx >= end_idx:
                break

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

    if chunk_entries:
        flush_chunk(chunk_entries, jsonl_file)
        print(f'  Saved progress: {global_idx} / {end_idx} '
              f'(unparseable so far: {none_count})')

print(f'\nGPU chunk complete. Entries {args.start_idx} to {end_idx} done.')
print(f'Unparseable: {none_count}')
print(f'Output: {jsonl_path}')
print(f'Run combine_llm_output.py once all GPUs are finished.')

if args.save_prompts > 0:
    _flush_examples()
