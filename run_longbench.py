import os
import argparse
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer
from COREGEN import LlamaForCausalLM
from tqdm import tqdm
import numpy as np
import random
import torch.multiprocessing as mp
import re


def get_first_line_not_comment(code: str, language: str = "python"):
    """
    This function gets the first line of code that is not a comment.

    Args:
    code: Str, the code

    Returns:
    Str, the first line of code that is not a comment or the first line of code if there is no line that is not a comment
    """

    # check if the language is valid
    assert language in ["python", "java"], "language must be one of [python, java]"

    # first remove the \n at the beginning of the code
    code = code.lstrip('\n')

    lines = code.split('\n')
    in_multiline_comment = False

    if language == "python":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('#'):
                continue
            # if the line is not a comment, then return the line
            return line

    elif language == "java":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and line.strip().startswith('/*'):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and line.strip().endswith('*/'):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('//'):
                continue
            # if the line is not a comment, then return the line
            return line

    # if we cannot find a line that is not a comment, then return the first line
    return lines[0]


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        try:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        except ImportError:
            print("Warning: fastchat not installed, using default prompt format")
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "cope" in model_name:
        prompt = prompt
    return prompt


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model_path,
             out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device)
    print(sum(p.numel() for p in model.parameters()))
    i = 0
    for json_obj in tqdm(data):
        if json_obj['language'] != 'python':
            continue
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if "chatglm3" in model_name:
            tokenized_prompt = \
            tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                           "repobench-p"]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            prompt = prompt.replace('    ', '\t')
            # print(prompt)
            input = tokenizer(prompt, return_tensors="pt").to(device)

            if input.input_ids.shape[1] > max_length:
                # Keep the last `context_length` tokens (truncate from the left/beginning)
                input = {
                    'input_ids': input.input_ids[:, -max_length:],
                    'attention_mask': input.attention_mask[:, -max_length:]
                }
            context_length = input['input_ids'].size(-1)
        if dataset == "samsum":  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                use_cache=True,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = get_first_line_not_comment(pred)
        print(pred)
        with open(out_path, "a", encoding="utf-8") as f:
            json_obj['answers'] = json_obj["answers"][0].replace("    ", "\t")
            json.dump({"idx": i, "pred": pred, "gt": json_obj["answers"], "all_classes": json_obj["all_classes"],
                       "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        i += 1


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_path, model_name, device):
    print(f"Loading model from: {model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        attn_implementation='eager',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model = model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run LongBench evaluation")
    parser.add_argument("--model_path", type=str, default="CODECOREGEN/COREGEN", help="Path to the model")
    parser.add_argument("--model_name", type=str, default="COREGEN", help="Model name")
    parser.add_argument("--max_length", type=int, default=6000, help="Maximum sequence length")
    parser.add_argument("--datasets", nargs="+", default=["lcc", "repobench-p"], help="Datasets to evaluate")
    parser.add_argument("--output_dir", type=str, default="./pred", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    seed_everything(args.seed)
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Predict on each dataset
    for dataset in args.datasets:
        print(f"Evaluating on dataset: {dataset}")
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        out_path = f"{args.output_dir}/{dataset}.jsonl"

        if dataset == 'lcc':
            prompt_format = "{context}"
        else:
            prompt_format = "{context}{input}"

        max_gen = 64
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []

        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], args.max_length,
                                                  max_gen, prompt_format, dataset, device, args.model_name,
                                                  args.model_path, out_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"Completed evaluation for {dataset}")


if __name__ == '__main__':
    main()
