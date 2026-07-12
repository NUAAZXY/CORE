"""
Unified evaluation script for COREGEN models.

Usage:
    python evaluate.py ppl     --model_path <path> [options]
    python evaluate.py repobench --model_path <path> [options]
    python evaluate.py longbench --model_path <path> [options]
    python evaluate.py metrics --pred_path <path> [options]
"""

import os
import argparse
import json
import re
import logging
from tqdm import tqdm

import torch
import numpy as np
import random
from torch.nn import CrossEntropyLoss
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, DatasetDict

try:
    from repo_eval.data.utils import construct_prompt
except ImportError:
    def construct_prompt(data, language="python"):
        context = data.get('context', '')
        return f"# Complete the following code:\n{context}"

try:
    from evaluation.metrics import exact_match_score, edit_similarity_score, codebleu_score
except ImportError:
    exact_match_score = edit_similarity_score = codebleu_score = None

logger = logging.getLogger(__name__)


# ============================================================
# Shared utilities
# ============================================================

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model_and_tokenizer(model_path, device="cuda"):
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation='eager',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()

    return model, tokenizer


def get_first_line_not_comment(code: str, language: str = "python"):
    assert language in ["python", "java"]
    code = code.lstrip('\n')
    lines = code.split('\n')
    in_multiline = False

    comment_start = ('"""', "'''") if language == "python" else ('/*',)
    comment_end = ('"""', "'''") if language == "python" else ('*/',)
    single_comment = '#' if language == "python" else '//'

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not in_multiline and any(stripped.startswith(s) for s in comment_start):
            in_multiline = True
            continue
        if in_multiline and any(stripped.endswith(s) for s in comment_end):
            in_multiline = False
            continue
        if in_multiline:
            continue
        if stripped.startswith(single_comment):
            continue
        return line

    return lines[0] if lines else ""


# ============================================================
# PPL evaluation
# ============================================================

class ConstantLengthDatasetExp(IterableDataset):
    def __init__(self, tokenizer, dataset, seq_length=8192):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length

    def __iter__(self):
        for item in self.dataset:
            content = item.get('content', '')
            if not content.strip():
                continue
            token_ids = self.tokenizer(content, truncation=False, add_special_tokens=False)['input_ids']
            for i in range(0, len(token_ids), self.seq_length):
                chunk = token_ids[i: i + self.seq_length]
                if len(chunk) == self.seq_length:
                    yield torch.tensor(chunk)


def run_ppl(args):
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    logger.info(f"Loading dataset from: {args.valid_dataset}")
    valid_data = load_from_disk(args.valid_dataset)
    dataset = ConstantLengthDatasetExp(tokenizer, valid_data, seq_length=args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model.eval()
    vocab_size = model.config.vocab_size
    losses = [0.0, 0.0, 0.0, 0.0]
    counts = [0, 0, 0, 0]
    corrects = [0, 0, 0, 0]
    val_len = [0, 1024, 2048, 4096, 8192]

    total_steps = 0
    for step, batch in enumerate(tqdm(dataloader, desc="PPL eval")):
        with torch.no_grad():
            batch = batch.to(args.device)
            outputs = model(batch, labels=batch, use_cache=False)

        for i in range(4):
            logits = outputs.logits[:, val_len[i]:val_len[i+1]-1].contiguous().view(-1, vocab_size)
            labels = batch[:, val_len[i]+1:val_len[i+1]].contiguous().view(-1).to(logits.device)
            pred = torch.argmax(logits, dim=-1)
            corrects[i] += (pred == labels).sum().item()
            counts[i] += logits.size(0)
            losses[i] += CrossEntropyLoss()(logits, labels).item()

        total_steps = step + 1
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break

    ranges = ["0-1024", "1024-2048", "2048-4096", "4096-8192"]
    print(f"\n{'Range':<14} {'Loss':<10} {'PPL':<12} {'Accuracy':<10}")
    print("-" * 46)
    results = []
    for i in range(4):
        avg_loss = losses[i] / total_steps
        ppl = np.exp(avg_loss)
        acc = corrects[i] / max(counts[i], 1)
        results.append((avg_loss, ppl, acc))
        print(f"{ranges[i]:<14} {avg_loss:<10.4f} {ppl:<12.4f} {acc:<10.4f}")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Dataset: {args.valid_dataset}\n")
            f.write(f"Seq Length: {args.seq_length}\n")
            f.write(f"Eval Steps: {total_steps}\n\n")
            f.write(f"{'Range':<14} {'Loss':<10} {'PPL':<12} {'Accuracy':<10}\n")
            for i in range(4):
                f.write(f"{ranges[i]:<14} {results[i][0]:<10.4f} {results[i][1]:<12.4f} {results[i][2]:<10.4f}\n")
        logger.info(f"Results saved to: {args.output_file}")


# ============================================================
# RepoBench evaluation
# ============================================================

def filter_dataset_by_levels(dataset: DatasetDict, levels: list) -> DatasetDict:
    return DatasetDict({
        name: subset.filter(lambda x: x['level'] in levels)
        for name, subset in dataset.items()
    })


def run_repobench(args):
    seed_everything(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataset = load_from_disk(args.dataset_path)
    dataset = filter_dataset_by_levels(dataset, args.levels)

    model_name = args.model_path.rstrip('/').split('/')[-1]
    save_dir = f"{args.output_dir}/{model_name}-{args.language}"
    os.makedirs(save_dir, exist_ok=True)

    tasks = ['cross_file_first', 'cross_file_random', 'in_file']
    for task in tasks:
        if task not in dataset:
            logger.warning(f"Task {task} not found in dataset, skipping")
            continue
        out_file = f"{save_dir}/{task}.jsonl"
        print(f"Evaluating task: {task} ({len(dataset[task])} samples)")

        for i in tqdm(range(0, len(dataset[task]), args.batch_size), desc=task):
            batch_data = [dataset[task][j] for j in range(i, min(i + args.batch_size, len(dataset[task])))]
            batch_prompts = [
                construct_prompt(d, language=args.language).replace("    ", "\t") for d in batch_data
            ]
            batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(args.device)

            if batch_inputs.input_ids.shape[1] > args.max_token_nums:
                batch_inputs = {
                    'input_ids': batch_inputs.input_ids[:, -args.max_token_nums:],
                    'attention_mask': batch_inputs.attention_mask[:, -args.max_token_nums:]
                }

            batch_outputs = model.generate(
                **batch_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True
            )

            for j, outputs in enumerate(batch_outputs):
                result = tokenizer.decode(
                    outputs[batch_inputs["input_ids"][j].shape[-1]:],
                    skip_special_tokens=True
                )
                result = get_first_line_not_comment(result, language=args.language)
                with open(out_file, "a") as f_out:
                    f_out.write(json.dumps({
                        "idx": i + j,
                        "level": batch_data[j]["level"],
                        "pred": result,
                        "gt": batch_data[j]["next_line"].replace("    ", "\t")
                    }) + "\n")

        print(f"Completed: {task}")

    print(f"\nPredictions saved to: {save_dir}")
    print(f"Run metrics: python evaluate.py metrics --pred_path {save_dir}")


# ============================================================
# LongBench evaluation
# ============================================================

def run_longbench(args):
    seed_everything(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    all_data = load_from_disk(args.dataset_path)

    for dataset_name in args.datasets:
        if dataset_name not in all_data:
            logger.warning(f"Dataset {dataset_name} not found, skipping")
            continue

        data = all_data[dataset_name]
        out_path = f"{args.output_dir}/{dataset_name}.jsonl"
        prompt_format = "{context}" if dataset_name == 'lcc' else "{context}{input}"

        print(f"Evaluating: {dataset_name}")
        i = 0
        for json_obj in tqdm(data, desc=dataset_name):
            if json_obj.get('language', 'python') != 'python':
                continue

            prompt = prompt_format.format(**json_obj).replace('    ', '\t')
            input_enc = tokenizer(prompt, return_tensors="pt").to(args.device)

            if input_enc.input_ids.shape[1] > args.max_length:
                input_enc = {
                    'input_ids': input_enc.input_ids[:, -args.max_length:],
                    'attention_mask': input_enc.attention_mask[:, -args.max_length:]
                }

            context_length = input_enc['input_ids'].size(-1)

            output = model.generate(
                **input_enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )[0]

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = get_first_line_not_comment(pred)

            with open(out_path, "a", encoding="utf-8") as f:
                answers = json_obj.get("answers", "")
                gt = answers[0] if isinstance(answers, list) else answers
                gt = gt.replace("    ", "\t")
                json.dump({
                    "idx": i,
                    "pred": pred,
                    "gt": gt,
                    "all_classes": json_obj.get("all_classes", []),
                    "length": json_obj.get("length", 0)
                }, f, ensure_ascii=False)
                f.write('\n')
            i += 1

        print(f"Completed {dataset_name}: {i} samples -> {out_path}")

    print(f"\nRun metrics: python evaluate.py metrics --pred_path {args.output_dir}")


# ============================================================
# Metrics computation (EM / Edit Similarity / CodeBLEU)
# ============================================================

def run_metrics(args):
    if exact_match_score is None:
        print("Error: evaluation.metrics module not found. Install it first.")
        return

    path = args.pred_path
    # Auto-detect format
    if os.path.exists(os.path.join(path, "cross_file_first.jsonl")):
        input_list = ["cross_file_first", "cross_file_random", "in_file"]
    else:
        input_list = ["lcc", "repobench-p"]

    # Separate data into rounds: same idx appearing again = next round
    # Round 1 = first occurrence, Round 2 = second occurrence, etc.
    # First pass: collect all data and detect number of rounds
    all_level_rounds = {}  # level -> {idx -> [entry_r1, entry_r2, ...]}
    global_num_rounds = 1
    for level in input_list:
        filepath = os.path.join(path, f"{level}.jsonl")
        if not os.path.exists(filepath):
            continue
        rounds = {}
        with open(filepath, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                idx = entry["idx"]
                if idx not in rounds:
                    rounds[idx] = []
                rounds[idx].append(entry)
        if rounds:
            all_level_rounds[level] = rounds
            global_num_rounds = max(global_num_rounds, max(len(v) for v in rounds.values()))

    # Compute and print metrics per round
    for r in range(global_num_rounds):
        if global_num_rounds > 1:
            print(f"\n--- Round {r+1} ---")
        total_points = 0
        total_em, total_es, total_cb = 0, 0, 0

        for level in input_list:
            if level not in all_level_rounds:
                print(f"  {level}: not found, skipping")
                continue
            rounds = all_level_rounds[level]
            data = []
            for idx in sorted(rounds.keys()):
                if r < len(rounds[idx]):
                    data.append(rounds[idx][r])
            if not data:
                continue

            ground_truth = [d["gt"] for d in data]
            generated = [d["pred"] for d in data]

            em = round(exact_match_score(ground_truth, generated) * 100, 2)
            es = round(edit_similarity_score(ground_truth, generated), 2)
            cb = round(codebleu_score(generated, ground_truth, args.language) * 100, 2)

            n = len(data)
            total_points += n
            total_em += em * n
            total_es += es * n
            total_cb += cb * n

            print(f"  {level} ({n} samples): EM={em}, ES={es}, CB={cb}")

        if total_points > 0:
            print(f"  Weighted Average ({total_points} total): "
                  f"EM={round(total_em/total_points, 2)}, "
                  f"ES={round(total_es/total_points, 2)}, "
                  f"CB={round(total_cb/total_points, 2)}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified COREGEN evaluation")
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # --- ppl ---
    p_ppl = subparsers.add_parser("ppl", help="Perplexity evaluation at multiple sequence lengths")
    p_ppl.add_argument("--model_path", type=str, required=True)
    p_ppl.add_argument("--valid_dataset", type=str, default="../datasets/starcoder_20Btokens_val")
    p_ppl.add_argument("--seq_length", type=int, default=8192)
    p_ppl.add_argument("--batch_size", type=int, default=1)
    p_ppl.add_argument("--max_eval_steps", type=int, default=2000)
    p_ppl.add_argument("--device", type=str, default="cuda")
    p_ppl.add_argument("--output_file", type=str, default=None)

    # --- repobench ---
    p_rb = subparsers.add_parser("repobench", help="RepoBench code completion evaluation")
    p_rb.add_argument("--model_path", type=str, required=True)
    p_rb.add_argument("--dataset_path", type=str, default="datasets/repobench")
    p_rb.add_argument("--max_token_nums", type=int, default=2000)
    p_rb.add_argument("--levels", nargs="+", default=["2k", "4k", "8k", "12k", "16k"])
    p_rb.add_argument("--language", type=str, default="python")
    p_rb.add_argument("--max_new_tokens", type=int, default=32)
    p_rb.add_argument("--batch_size", type=int, default=1)
    p_rb.add_argument("--output_dir", type=str, default="./results_repo")
    p_rb.add_argument("--device", type=str, default="cuda")
    p_rb.add_argument("--seed", type=int, default=42)

    # --- longbench ---
    p_lb = subparsers.add_parser("longbench", help="LongBench evaluation (lcc, repobench-p)")
    p_lb.add_argument("--model_path", type=str, required=True)
    p_lb.add_argument("--dataset_path", type=str, default="datasets/LongBench")
    p_lb.add_argument("--datasets", nargs="+", default=["lcc", "repobench-p"])
    p_lb.add_argument("--max_length", type=int, default=6000)
    p_lb.add_argument("--max_new_tokens", type=int, default=64)
    p_lb.add_argument("--output_dir", type=str, default="./pred_longbench")
    p_lb.add_argument("--device", type=str, default="cuda")
    p_lb.add_argument("--seed", type=int, default=42)

    # --- metrics ---
    p_m = subparsers.add_parser("metrics", help="Compute EM/ES/CodeBLEU from prediction files")
    p_m.add_argument("--pred_path", type=str, required=True)
    p_m.add_argument("--language", type=str, default="python")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.mode is None:
        parser.print_help()
        return

    dispatch = {
        "ppl": run_ppl,
        "repobench": run_repobench,
        "longbench": run_longbench,
        "metrics": run_metrics,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
