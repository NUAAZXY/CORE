import argparse
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset
from torch.nn import CrossEntropyLoss
from pathlib import Path
import logging

from transformers import LlamaTokenizer
from COREGEN import LlamaForCausalLM
from datasets import load_from_disk



class ConstantLengthDatasetExp(IterableDataset):
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024,
                 drop_last=True, add_special_tokens=False):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.drop_last = drop_last
        self.add_special_tokens = add_special_tokens
        self.epoch = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True

        while more_examples:
            try:
                item = next(iterator)
                # Add error checking
                if 'content' not in item:
                    continue

                content = item['content']
                if not content.strip():  # Skip empty content
                    continue

                # Tokenize with proper parameters
                tokenized_input = self.tokenizer(
                    content,
                    truncation=False,
                    add_special_tokens=self.add_special_tokens
                )['input_ids']

                # Process in chunks
                for i in range(0, len(tokenized_input), self.seq_length):
                    input_ids = tokenized_input[i: i + self.seq_length]

                    if len(input_ids) == self.seq_length:
                        yield torch.tensor(input_ids)
                    elif not self.drop_last and len(input_ids) > 0:
                        # If not dropping last chunk, pad it
                        padded = input_ids + [self.tokenizer.pad_token_id] * (self.seq_length - len(input_ids))
                        yield torch.tensor(padded)

            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                    self.epoch += 1
                    print(f"Starting epoch {self.epoch}")
                else:
                    more_examples = False
            except Exception as e:
                print(f"Error processing item: {e}")
                continue  # Skip problematic items


def create_dataloaders(tokenizer, args):
    """Create evaluation dataloader"""
    valid_data = load_from_disk(args.valid_dataset)
    valid_dataset = ConstantLengthDatasetExp(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        drop_last=True,
        add_special_tokens=False
    )
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    return eval_dataloader


def evaluate_extrapolation(model, eval_dataloader, args):
    """Evaluate model on different sequence lengths"""
    model.eval()
    losses = [0, 0, 0, 0]
    counts = [0, 0, 0, 0]
    corrects = [0, 0, 0, 0]
    val_len = [0, 1024, 2048, 4096, 8192]

    total_steps = 0

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = batch.to(args.device)
            outputs = model(batch, labels=batch, use_cache=False)

        for i in range(len(val_len) - 1):
            logits = outputs.logits[:, val_len[i]:val_len[i + 1] - 1].contiguous().view(-1, model.config.vocab_size)
            labels = batch[:, val_len[i] + 1: val_len[i + 1]].contiguous().view(-1).to(logits.device)
            pred = torch.argmax(logits, dim=-1)
            corrects[i] += (pred.squeeze() == labels).tolist().count(True)
            counts[i] += logits.size(0)

            loss_fn = CrossEntropyLoss()
            losses[i] += loss_fn(logits, labels)

        total_steps = step + 1
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break

    # Calculate average losses
    losses = [l / total_steps for l in losses]

    try:
        perplexity = [torch.exp(loss) for loss in losses]
    except OverflowError:
        perplexity = [float("inf") for i in range(len(corrects))]

    accuracies = [corrects[i] / counts[i] for i in range(len(corrects))]

    return losses, perplexity, accuracies


def download_model(model_name, cache_dir=None):
    """Download model from HuggingFace"""
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            attn_implementation='eager',
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir
        )
        return model
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate language model perplexity")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="CODECOREGEN/COREGEN",
                        help="Model name or path")
    parser.add_argument("--local_model_path", type=str, default=None,
                        help="Local model path (if not downloading)")
    parser.add_argument("--tokenizer_name", type=str, default="CODECOREGEN/COREGEN",
                        help="Tokenizer name")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for downloaded models")

    # Data parameters
    parser.add_argument("--valid_dataset", type=str, default="datasets/starcoder_20Btokens_val",
                        help="Validation dataset path")
    parser.add_argument("--seq_length", type=int, default=8192,
                        help="Sequence length for evaluation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size")

    # Evaluation parameters
    parser.add_argument("--max_eval_steps", type=int, default=2000,
                        help="Maximum evaluation steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation")

    # Other parameters
    parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt",
                        help="Output file for results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name)

    # Create dataloader
    logger.info(f"Creating dataloader for dataset: {args.valid_dataset}")
    eval_dataloader = create_dataloaders(tokenizer, args)

    # Load model
    if args.local_model_path:
        logger.info(f"Loading local model from: {args.local_model_path}")
        model = LlamaForCausalLM.from_pretrained(
            args.local_model_path,
            attn_implementation='eager',
            torch_dtype=torch.bfloat16
        )
    else:
        logger.info(f"Downloading model: {args.model_name}")
        model = download_model(args.model_name, args.cache_dir)
        if model is None:
            logger.error("Failed to download model")
            return

    # Move model to device
    model = model.to(args.device)

    # Enable gradient checkpointing if requested
    if args.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info("Starting evaluation...")
    eval_loss, perplexity, accuracy = evaluate_extrapolation(model, eval_dataloader, args)

    # Format results
    eval_loss_formatted = [round(e.item(), 2) for e in eval_loss]
    perplexity_formatted = [round(p.item(), 2) for p in perplexity]
    accuracy_formatted = [round(a, 2) for a in accuracy]

    # Print results
    print(f"Evaluation Results:")
    print(f"Loss: {eval_loss_formatted}")
    print(f"Perplexity: {perplexity_formatted}")
    print(f"Accuracy: {accuracy_formatted}")

    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write(f"Evaluation Results:\n")
        f.write(f"Model: {args.model_name if not args.local_model_path else args.local_model_path}\n")
        f.write(f"Dataset: {args.valid_dataset}\n")
        f.write(f"Sequence Length: {args.seq_length}\n")
        f.write(f"Max Eval Steps: {args.max_eval_steps}\n")
        f.write(f"Loss (1024, 2048, 4096, 8192): {eval_loss_formatted}\n")
        f.write(f"Perplexity (1024, 2048, 4096, 8192): {perplexity_formatted}\n")
        f.write(f"Accuracy (1024, 2048, 4096, 8192): {accuracy_formatted}\n")

    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()