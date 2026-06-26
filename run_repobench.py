import os
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import LlamaTokenizer
from COREGEN_test import LlamaForCausalLM
from datasets import DatasetDict, Dataset
import pandas as pd
import torch

# Import repo_eval utilities if available
try:
    from repo_eval.data.utils import construct_prompt
except ImportError:
    print("Warning: repo_eval not found. Using fallback prompt construction.")


    def construct_prompt(data, language="python"):
        # Fallback implementation
        context = data.get('context', '')
        return f"# Complete the following code:\n{context}"


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


def filter_dataset_by_date_range(dataset: DatasetDict, start_date: str, end_date: str) -> DatasetDict:
    """
    Filters a Huggingface dataset by a specific date range.

    Parameters:
    dataset (DatasetDict): The input dataset with subsets containing a 'created_at' column.
    start_date (str): The start date in the format 'YYYY-MM-DD'.
    end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
    DatasetDict: The filtered dataset.
    """
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')

    filtered_dataset_dict = {}

    for subset_name in dataset.keys():
        df = pd.DataFrame(dataset[subset_name])
        df['created_at'] = pd.to_datetime(df['created_at'])

        # Filter the DataFrame
        mask = (df['created_at'] >= start_date) & (df['created_at'] <= end_date)
        filtered_df = df[mask]

        # Convert back to Huggingface Dataset
        filtered_dataset_dict[subset_name] = Dataset.from_pandas(filtered_df)

    return DatasetDict(filtered_dataset_dict)


def filter_dataset_by_levels(dataset: DatasetDict, levels: list) -> DatasetDict:
    """
    Filters a Huggingface dataset by specific levels.

    Parameters:
    dataset (DatasetDict): The input dataset with subsets containing a 'level' column.
    levels (list): The list of levels to filter by.

    Returns:
    DatasetDict: The filtered dataset.
    """
    filtered_dataset_dict = {}

    for subset_name in dataset.keys():
        # Filter the subset directly using the 'filter' method
        filtered_subset = dataset[subset_name].filter(lambda example: example['level'] in levels)
        filtered_dataset_dict[subset_name] = filtered_subset

    return DatasetDict(filtered_dataset_dict)


def main():
    parser = argparse.ArgumentParser(description="Run RepoBench evaluation")
    parser.add_argument("--model_path", type=str, default="CODECOREGEN/COREGEN", help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default="tianyang/repobench_python_v1.1", help="Path to the dataset")
    parser.add_argument("--max_token_nums", type=int, default=6000, help="Max token number for the prompt")
    parser.add_argument("--levels", nargs="+", default=["2k", "4k", "8k", "12k", "16k"], help="Levels to filter by")
    parser.add_argument("--language", type=str, default="python", help="Programming language")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")

    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_path)


    # Filter the dataset by levels
    dataset = filter_dataset_by_levels(dataset, args.levels)

    # Load the model and tokenizer
    print(f"Loading model from: {args.model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation='eager',
        torch_dtype=torch.bfloat16
    ).eval().cuda()

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    # Create the save directory
    model_name = args.model_path.split('/')[-1]
    save_dir = f"{args.output_dir}/{model_name}-{args.language}"
    os.makedirs(save_dir, exist_ok=True)

    tasks = ['cross_file_first', 'cross_file_random', 'in_file']
    # print(dataset['train'])
    for task in tasks:
        print(f"Evaluating task: {task}")
        for i in tqdm(range(0, len(dataset[task]), args.batch_size), desc=f"Generating {task}"):
            batch_data = [dataset[task][j] for j in range(i, min(i + args.batch_size, len(dataset[task])))]
            batch_prompts = [
                construct_prompt(d, language=args.language).replace("    ", "\t") for d in batch_data
            ]

            batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

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

                with open(f"{save_dir}/{task}.jsonl", "a") as f_out:
                    f_out.write(json.dumps({
                        "idx": i + j,
                        "level": batch_data[j]["level"],
                        "pred": result,
                        "gt": batch_data[j]["next_line"].replace("    ", "\t")
                    }) + "\n")

        print(f"Completed task: {task}")


if __name__ == "__main__":
    main()
