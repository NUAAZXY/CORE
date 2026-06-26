import argparse
import os
import logging
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from transformers import LlamaTokenizer, LlamaConfig
from COREGEN import LlamaForCausalLM
from transformers import AdamW, get_scheduler, set_seed
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
import datasets
import transformers


class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)['content'])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logging.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            if buffer:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)['input_ids']
                all_token_ids = []
                for tokenized_input in tokenized_inputs:
                    all_token_ids.extend(tokenized_input + [self.concat_token_id])
                for i in range(0, len(all_token_ids), self.seq_length):
                    input_ids = all_token_ids[i: i + self.seq_length]
                    if len(input_ids) == self.seq_length:
                        yield torch.tensor(input_ids)


class ConstantLengthDatasetExp(IterableDataset):
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True

        while more_examples:
            try:
                item = next(iterator)['content']
                tokenized_input = self.tokenizer(item, truncation=False)['input_ids']
                for i in range(0, len(tokenized_input), self.seq_length):
                    input_ids = tokenized_input[i: i + self.seq_length]
                    if len(input_ids) == self.seq_length:
                        yield torch.tensor(input_ids)

            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                    self.epoch += 1
                    logging.info(f"Dataset epoch: {self.epoch}")
                else:
                    more_examples = False
                    break


def setup_logging(accelerator, project_name, args):
    logger = logging.getLogger(__name__)

    # Create log directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_dir / f"debug_{accelerator.process_index}.log"),
            logging.StreamHandler()
        ]
    )

    if accelerator.is_main_process:
        if args.use_wandb:
            wandb.init(project=project_name, config=vars(args))
            run_name = wandb.run.name
        else:
            run_name = 'local_run'

        tb_writer = SummaryWriter(log_dir / "tensorboard")
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return logger, tb_writer, run_name


def create_dataloaders(tokenizer, args):
    train_data = load_from_disk(args.train_dataset)
    valid_data = load_from_disk(args.valid_dataset)

    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=args.seq_length
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length
    )
    valid_extrapolate_dataset = ConstantLengthDatasetExp(
        tokenizer, valid_data, infinite=False, seq_length=args.extrapolate_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    eval_extrapolate_dataloader = DataLoader(valid_extrapolate_dataset, batch_size=args.valid_batch_size)

    return train_dataloader, eval_dataloader, eval_extrapolate_dataloader


def get_grouped_params(model, args, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {'params': params_with_wd, 'weight_decay': args.weight_decay},
        {'params': params_without_wd, 'weight_decay': 0.0}
    ]


def log_metrics(accelerator, logger, tb_writer, step, metrics, use_wandb):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        if use_wandb:
            wandb.log(metrics)
        if tb_writer:
            [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]


def evaluate(model, eval_dataloader, accelerator, args):
    model.eval()
    losses = []
    count = 0
    correct = 0

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch, use_cache=False)

        logits = outputs.logits[:, :-1].contiguous().view(-1, model.config.vocab_size)
        labels = batch[:, 1:].contiguous().view(-1).to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        correct += (pred.squeeze() == labels).tolist().count(True)
        count += logits.size(0)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break

    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return loss.item(), perplexity.item(), correct / count


def evaluate_extrapolation(model, eval_extrapolation_dataloader, args):
    model.eval()
    losses = [0, 0, 0, 0]
    counts = [0, 0, 0, 0]
    corrects = [0, 0, 0, 0]
    val_len = [0, 1024, 2048, 4096, 8192]

    for step, batch in enumerate(eval_extrapolation_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch, use_cache=False)

        for i in range(len(val_len) - 1):
            logits = outputs.logits[:, val_len[i]:val_len[i + 1] - 1].contiguous().view(-1, model.config.vocab_size)
            labels = batch[:, val_len[i] + 1: val_len[i + 1]].contiguous().view(-1).to(logits.device)
            pred = torch.argmax(logits, dim=-1)
            corrects[i] += (pred.squeeze() == labels).tolist().count(True)
            counts[i] += logits.size(0)
            losses[i] += torch.mean(CrossEntropyLoss()(logits, labels).view(-1))

        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break

    losses = [l / step for l in losses]
    try:
        perplexity = [torch.exp(loss) for loss in losses]
    except OverflowError:
        perplexity = [float("inf") for i in range(len(corrects))]

    return losses, perplexity, [corrects[i] / counts[i] for i in range(len(corrects))]


def parse_args():
    parser = argparse.ArgumentParser(description="Train language model with COPE")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="codellama-7b", help="Base model name")
    parser.add_argument("--vocab_size", type=int, default=32016, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Intermediate size")
    parser.add_argument("--max_position_embeddings", type=int, default=32768, help="Max position embeddings")

    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--num_warmup_steps", type=int, default=3000, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_train_steps", type=int, default=150000, help="Maximum training steps")
    parser.add_argument("--max_eval_steps", type=int, default=50, help="Maximum evaluation steps")

    # Data parameters
    parser.add_argument("--train_dataset", type=str, default="starcoder_20B", help="Training dataset path")
    parser.add_argument("--valid_dataset", type=str, default="starcoder_20Btokens_val", help="Validation dataset path")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--extrapolate_length", type=int, default=8192, help="Extrapolation length")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_checkpoint_steps", type=int, default=5000, help="Save checkpoint steps")
    parser.add_argument("--log_step", type=int, default=5000, help="Log steps")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--project_name", type=str, default="COREGEN", help="Project name")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variables
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "offline"

    # Initialize accelerator
    accelerator = Accelerator()

    # Add accelerator state to args
    acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
    for k, v in acc_state.items():
        setattr(args, k, v)

    samples_per_step = accelerator.state.num_processes * args.train_batch_size
    set_seed(args.seed)

    # Setup logging
    logger, tb_writer, run_name = setup_logging(accelerator, args.project_name, args)
    logger.info(f"Accelerator state: {accelerator.state}")

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)

    # Create model configuration
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        attn_implementation='eager'
    )

    # Initialize model from scratch
    model = LlamaForCausalLM(config)
    logger.info(f'Model parameters: {sum(x.numel() for x in model.parameters())}')

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Create dataloaders
    train_dataloader, eval_dataloader, eval_extrapolation_dataloader = create_dataloaders(tokenizer, args)

    # Setup optimizer and scheduler
    optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    def get_lr():
        return optimizer.param_groups[0]['lr']

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Training loop
    model.train()
    completed_steps = 0

    for step, batch in enumerate(tqdm(train_dataloader,
                                      total=args.max_train_steps * args.gradient_accumulation_steps,
                                      leave=False)):
        outputs = model(batch, labels=batch, use_cache=False)
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        if step % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        if step % args.save_checkpoint_steps == 0 and step > 0:
            logger.info('Evaluating and saving model checkpoint')
            eval_loss_, perplexity_, acc_ = evaluate_extrapolation(model, eval_extrapolation_dataloader, args)

            metrics = {
                'lr': get_lr(),
                'samples': step * samples_per_step,
                'steps': completed_steps,
                'loss/train': loss.item(),
                'loss/eval_1024': eval_loss_[0].item(),
                'ppl/perplexity_1024': perplexity_[0].item(),
                'acc/acc_1024': acc_[0],
                'loss/eval_2048': eval_loss_[1].item(),
                'ppl/perplexity_2048': perplexity_[1].item(),
                'acc/acc_2048': acc_[1],
                'loss/eval_4096': eval_loss_[2].item(),
                'ppl/perplexity_4096': perplexity_[2].item(),
                'acc/acc_4096': acc_[2],
                'loss/eval_8192': eval_loss_[3].item(),
                'ppl/perplexity_8192': perplexity_[3].item(),
                'acc/acc_8192': acc_[3]
            }

            log_metrics(accelerator, logger, tb_writer, step, metrics, args.use_wandb)

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = output_dir / f"checkpoint_{step}"
            unwrapped_model.save_pretrained(save_path, save_function=accelerator.save)

            model.train()

        if completed_steps >= args.max_train_steps:
            break

    # Save final checkpoint
    logger.info('Evaluating and saving final model checkpoint')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    final_save_path = output_dir / f"final_checkpoint_{step}"
    unwrapped_model.save_pretrained(final_save_path, save_function=accelerator.save)

    if accelerator.is_main_process and tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
