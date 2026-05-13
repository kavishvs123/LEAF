"""
finetune_llm_qlora.py

Fine-tunes LLaMA on traffic prediction selection using QLoRA (4-bit quantization
+ LoRA). Uses {dataset}_choices.json where optimal_idx is the ground truth label
for which candidate prediction is best.

QLoRA loads the model in 4-bit precision (~4GB), dramatically reducing VRAM
usage compared to standard LoRA. Recommended when GPU memory is limited.

The fine-tuned LoRA adapter is saved to --output_dir. To then use it for
inference, pass --model_path and --lora_path to generate_llm_output.py.

Requirements:
    pip install trl peft bitsandbytes

Usage:
    python finetune_llm_qlora.py --dataset PEMS08
    python finetune_llm_qlora.py --dataset PEMS08 --num_samples 50000 --num_epochs 1
"""

import argparse
import json
import os
import random
import torch
import ijson
import pandas as pd
from datasets import Dataset

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS08'])
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--dump_dir', type=str, default='./outputs/dump')
parser.add_argument('--output_dir', type=str, default='./outputs/qlora',
                    help='Where to save the QLoRA adapter weights')
parser.add_argument('--num_samples', type=int, default=500000,
                    help='Number of entries to sample from choices file for training')
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4,
                    help='Per-device training batch size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                    help='Accumulate gradients over N steps before updating weights')
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--lora_r', type=int, default=16,
                    help='LoRA rank — higher = more parameters, more capacity')
parser.add_argument('--lora_alpha', type=int, default=32,
                    help='LoRA scaling factor — typically 2x lora_r')
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--max_seq_len', type=int, default=2048,
                    help='Max token length for training examples')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)


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

RESPONSE_TEMPLATE = '### Selected candidate: '


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


def build_training_text(entry: dict) -> str:
    """Full training text = prompt + response template + ground truth answer.

    The newline before the template is kept in the text but excluded from
    RESPONSE_TEMPLATE itself so DataCollatorForCompletionOnlyLM can find
    the token sequence reliably (leading \\n shifts token boundaries).
    """
    prompt      = build_prompt(entry)
    answer      = str(entry['optimal_idx'] + 1)  # convert 0-indexed to 1-indexed
    return prompt + '\n' + RESPONSE_TEMPLATE + answer


# ── Sample entries from choices file using reservoir sampling ─────────────────

choices_path = os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices{args.postfix}.json')
print(f'Sampling {args.num_samples} entries from {choices_path}...')

# Reservoir sampling — single pass through the file, O(num_samples) memory
reservoir = []
with open(choices_path, 'rb') as f:
    for i, entry in enumerate(ijson.items(f, 'item')):
        if len(reservoir) < args.num_samples:
            reservoir.append(entry)
        else:
            j = random.randint(0, i)
            if j < args.num_samples:
                reservoir[j] = entry

print(f'Sampled {len(reservoir)} entries.')

# Build training texts
print('Building training examples...')
texts = [build_training_text(e) for e in reservoir]
del reservoir  # free memory before loading model

dataset = Dataset.from_dict({'text': texts})
print(f'Dataset: {len(dataset)} examples')


# ── Load model with 4-bit quantization (QLoRA) ────────────────────────────────

print(f'Loading model: {args.model_path}')
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'right'


# ── Apply LoRA ────────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=args.lora_dropout,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── Data collator — only compute loss on the answer, not the prompt ───────────

collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
)


# ── Training ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    bf16=True,
    fp16=False,
    logging_steps=100,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=2,         # keep only the 2 most recent checkpoints
    warmup_steps=500,
    lr_scheduler_type='cosine',
    report_to='none',
    seed=args.seed,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    dataset_text_field='text',
    max_seq_length=args.max_seq_len,
    tokenizer=tokenizer,
)

print('Starting fine-tuning...')
# resume_from_checkpoint=True automatically picks up the latest checkpoint
# in output_dir if one exists, otherwise starts fresh
trainer.train(resume_from_checkpoint=True)


# ── Save LoRA adapter ─────────────────────────────────────────────────────────

adapter_path = os.path.join(args.output_dir, f'{args.dataset.lower()}_qlora_adapter')
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print(f'\nDone. LoRA adapter saved to: {adapter_path}')
print(f'To use for inference, pass --lora_path {adapter_path} to generate_llm_output.py')
