import argparse
import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer
from transformers import AdamW, get_scheduler, set_seed
from datasets import load_from_disk
from accelerate import Accelerator
import datasets
import transformers
from COREGEN_qwen import Qwen2ForCausalLM as COREGENQwen2ForCausalLM
from torch.nn.utils import parametrize


# ======================== LoRA ========================

class LoRAParametrization(nn.Module):
    def __init__(self, in_features, out_features, rank=32, alpha=32):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, weight):
        lora_A = self.lora_A.to(weight.dtype)
        lora_B = self.lora_B.to(weight.dtype)
        return weight + (lora_B @ lora_A) * self.scaling


def inject_lora(model, target_layers_start=0, rank=32, alpha=32,
                target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj')):
    lora_count = 0
    for name, module in model.named_modules():
        if 'layers' in name and isinstance(module, nn.Linear):
            parts = name.split('.')
            try:
                layer_idx = int(parts[parts.index('layers') + 1])
            except (ValueError, IndexError):
                continue
            if layer_idx >= target_layers_start and parts[-1] in target_modules:
                parametrize.register_parametrization(module, "weight", LoRAParametrization(
                    module.in_features, module.out_features, rank, alpha))
                lora_count += 1
                logging.info(f"LoRA -> {name} (layer {layer_idx})")
    logging.info(f"Total LoRA modules: {lora_count}")
    return lora_count


class LoRAModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    def forward(self, input_ids, labels=None, use_cache=False, **kwargs):
        return self.base_model(input_ids, labels=labels, use_cache=use_cache, **kwargs)

    def save_pretrained(self, save_path, merge_lora=True, destructive=False, **kwargs):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_function = kwargs.get('save_function', torch.save)

        if merge_lora and destructive:
            for name, module in self.base_model.named_modules():
                if parametrize.is_parametrized(module, "weight"):
                    parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            self.base_model.save_pretrained(save_path, save_function=save_function)
        elif merge_lora:
            param_info = []
            for name, module in self.base_model.named_modules():
                if parametrize.is_parametrized(module, "weight"):
                    merged_weight = module.weight.detach().clone()
                    param_list = list(module.parametrizations.weight)
                    param_info.append((module, name, param_list, merged_weight))
            for module, name, param_list, merged_weight in param_info:
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            self.base_model.save_pretrained(save_path, save_function=save_function)
            for module, name, param_list, merged_weight in param_info:
                for lora_param in param_list:
                    lora_A = lora_param.lora_A.to(merged_weight.dtype)
                    lora_B = lora_param.lora_B.to(merged_weight.dtype)
                    original_weight = merged_weight - (lora_B @ lora_A) * lora_param.scaling
                    module.weight = nn.Parameter(original_weight, requires_grad=False)
                    parametrize.register_parametrization(module, "weight", lora_param)
        else:
            self.base_model.save_pretrained(save_path, save_function=save_function)
        logging.info(f"Saved model to {save_path}")


# ======================== Dataset ========================

class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=900,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
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
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=8192,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
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


# ======================== Logging & Eval ========================

def setup_logging(accelerator, project_name, args):
    logger = logging.getLogger(__name__)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
        handlers=[
            logging.FileHandler(log_dir / f"debug_qwen_s{args.stage}_{accelerator.process_index}.log"),
            logging.StreamHandler()
        ]
    )
    if accelerator.is_main_process:
        if args.use_wandb:
            wandb.init(project=project_name, config=vars(args))
        tb_writer = SummaryWriter(log_dir / f"tensorboard_qwen_s{args.stage}")
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer


def log_metrics(accelerator, logger, tb_writer, step, metrics, use_wandb):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        if use_wandb:
            wandb.log(metrics)
        if tb_writer:
            [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]


def evaluate(model, eval_dataloader, accelerator, args):
    model.eval()
    losses, count, correct = [], 0, 0
    vocab_size = accelerator.unwrap_model(model).config.vocab_size

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch, use_cache=False)
        logits = outputs.logits[:, :-1].contiguous().view(-1, vocab_size)
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


def evaluate_extrapolation(model, eval_extrapolation_dataloader, accelerator, args):
    model.eval()
    vocab_size = accelerator.unwrap_model(model).config.vocab_size
    losses = [0, 0, 0, 0]
    counts = [0, 0, 0, 0]
    corrects = [0, 0, 0, 0]
    val_len = [0, 1024, 2048, 4096, 8192]

    for step, batch in enumerate(eval_extrapolation_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch, use_cache=False)
        for i in range(len(val_len) - 1):
            logits = outputs.logits[:, val_len[i]:val_len[i + 1] - 1].contiguous().view(-1, vocab_size)
            labels = batch[:, val_len[i] + 1: val_len[i + 1]].contiguous().view(-1).to(logits.device)
            pred = torch.argmax(logits, dim=-1)
            corrects[i] += (pred.squeeze() == labels).tolist().count(True)
            counts[i] += logits.size(0)
            losses[i] += torch.mean(CrossEntropyLoss()(logits, labels).view(-1))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break

    num_steps = max(step, 1)
    losses = [l / num_steps for l in losses]
    try:
        perplexity = [torch.exp(loss) for loss in losses]
    except OverflowError:
        perplexity = [float("inf") for _ in range(len(corrects))]
    return losses, perplexity, [corrects[i] / counts[i] for i in range(len(corrects))]


# ======================== Args ========================

def parse_args():
    parser = argparse.ArgumentParser(description="COREGEN two-stage training for Qwen2.5-Coder-7B")

    # Stage control
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Stage 1: train DependencyEncoding (no mask, no LoRA). "
                             "Stage 2: LoRA finetune with block mask.")

    # Model
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model path. Stage 1 default: ~/models/Qwen2.5-Coder-7B. "
                             "Stage 2 default: ./results_qwen_s1/final_checkpoint")

    # LoRA (stage 2 only)
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (stage 2)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (stage 2)")
    parser.add_argument("--lora_target_layers_start", type=int, default=0, help="Starting layer for LoRA")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size per GPU")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=3000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Default: stage1=34722 (250M/900/8), stage2=3472 (25M/900/8)")
    parser.add_argument("--max_eval_steps", type=int, default=0)

    # Data
    parser.add_argument("--train_dataset", type=str, default="../starcoder_20Btokens")
    parser.add_argument("--valid_dataset", type=str, default="../datasets/starcoder_20Btokens_val")
    parser.add_argument("--seq_length", type=int, default=900, help="Sequence length (coupling tokens)")
    parser.add_argument("--extrapolate_length", type=int, default=8192)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_checkpoint_steps", type=int, default=5000)
    parser.add_argument("--log_step", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: stage1=./results_qwen_s1, stage2=./results_qwen_s2")
    parser.add_argument("--project_name", type=str, default="COREGEN-Qwen")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()

    # Apply stage-specific defaults
    if args.model_name is None:
        args.model_name = (os.path.expanduser("~/models/Qwen2.5-Coder-7B") if args.stage == 1
                           else "./results_qwen_s1/final_checkpoint")
    if args.max_train_steps is None:
        args.max_train_steps = 34722 if args.stage == 1 else 3472
    if args.output_dir is None:
        args.output_dir = f"./results_qwen_s{args.stage}"

    return args


# ======================== Main ========================

def main():
    args = parse_args()

    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "offline"

    accelerator = Accelerator()
    acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
    for k, v in acc_state.items():
        setattr(args, k, v)

    samples_per_step = accelerator.state.num_processes * args.train_batch_size
    set_seed(args.seed)
    logger, tb_writer = setup_logging(accelerator, args.project_name, args)
    logger.info(f"=== COREGEN Stage {args.stage} ===")
    logger.info(f"Accelerator state: {accelerator.state}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load COREGEN Qwen model
    base_model = COREGENQwen2ForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation='eager')

    if args.stage == 1:
        # Stage 1: train DependencyEncoding only, no block mask, no LoRA
        base_model.model.use_block_mask = False

        for param in base_model.parameters():
            param.requires_grad = False

        de_count = 0
        for name, module in base_model.named_modules():
            if 'DependencyEncoding' in type(module).__name__:
                for pn, param in module.named_parameters():
                    param.requires_grad = True
                    de_count += param.numel()
                    logging.info(f"Unfroze DE: {name}.{pn}, shape: {param.shape}")
        if de_count == 0:
            raise ValueError("No DependencyEncoding parameters found!")
        logging.info(f"DependencyEncoding trainable params: {de_count:,}")

        model = base_model  # no LoRA wrapper needed

    else:
        # Stage 2: LoRA finetune with block mask enabled, DE frozen
        base_model.model.use_block_mask = True

        for param in base_model.parameters():
            param.requires_grad = False

        logger.info(f"Injecting LoRA (rank={args.lora_rank}) into layers >= {args.lora_target_layers_start}")
        inject_lora(base_model,
                     target_layers_start=args.lora_target_layers_start,
                     rank=args.lora_rank, alpha=args.lora_alpha)

        model = LoRAModel(base_model)

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Dataloaders
    train_data = load_from_disk(args.train_dataset)
    valid_data = load_from_disk(args.valid_dataset)
    train_dataloader = DataLoader(
        ConstantLengthDataset(tokenizer, train_data, infinite=True, seq_length=args.seq_length),
        batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(
        ConstantLengthDataset(tokenizer, valid_data, infinite=False, seq_length=args.seq_length),
        batch_size=args.valid_batch_size)

    if args.stage == 2:
        eval_extrapolation_dataloader = DataLoader(
            ConstantLengthDatasetExp(tokenizer, valid_data, infinite=False, seq_length=args.extrapolate_length),
            batch_size=args.valid_batch_size)

    # Optimizer
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW([{'params': trainable_param_list, 'weight_decay': args.weight_decay}], lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    # Prepare with accelerator
    if args.stage == 2:
        model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader)
    else:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Training loop
    model.train()
    completed_steps = 0

    for step, batch in enumerate(tqdm(train_dataloader, total=args.max_train_steps, leave=False)):
        outputs = model(batch, labels=batch, use_cache=False)
        loss = outputs.loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        if step % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        if step % args.log_step == 0 and step > 0:
            log_metrics(accelerator, logger, tb_writer, step,
                        {'lr': get_lr(), 'loss/train': loss.item(), 'steps': completed_steps},
                        args.use_wandb)

        if step % args.save_checkpoint_steps == 0 and step > 0:
            logger.info('Evaluating and saving checkpoint')
            eval_loss, eval_ppl, eval_acc = evaluate(model, eval_dataloader, accelerator, args)
            metrics = {
                'lr': get_lr(), 'samples': step * samples_per_step, 'steps': completed_steps,
                'loss/train': loss.item(), 'loss/eval': eval_loss,
                'perplexity/eval': eval_ppl, 'accuracy/eval': eval_acc,
            }
            log_metrics(accelerator, logger, tb_writer, step, metrics, args.use_wandb)

            accelerator.wait_for_everyone()
            save_path = output_dir / f"checkpoint_{step}"
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                if args.stage == 1:
                    unwrapped.save_pretrained(save_path, save_function=accelerator.save)
                else:
                    unwrapped.save_pretrained(save_path, merge_lora=True, destructive=False,
                                              save_function=accelerator.save)
                tokenizer.save_pretrained(save_path)
            accelerator.wait_for_everyone()
            model.train()

        if completed_steps >= args.max_train_steps:
            break

    # Final save
    logger.info('Saving final checkpoint')
    eval_loss, eval_ppl, eval_acc = evaluate(model, eval_dataloader, accelerator, args)
    logger.info(f"Final eval - loss: {eval_loss:.4f}, ppl: {eval_ppl:.4f}, acc: {eval_acc:.4f}")

    accelerator.wait_for_everyone()
    final_save_path = output_dir / "final_checkpoint"
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if args.stage == 1:
            unwrapped.save_pretrained(final_save_path, save_function=accelerator.save)
        else:
            unwrapped.save_pretrained(final_save_path, merge_lora=True, destructive=True,
                                      save_function=accelerator.save)
        tokenizer.save_pretrained(final_save_path)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
