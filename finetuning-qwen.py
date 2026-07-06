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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import AdamW, get_scheduler, set_seed
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
import datasets
import transformers
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'


# LoRA via module replacement - replaces nn.Linear with a custom module
# that stores the frozen weight as a buffer and LoRA params as trainable parameters.
# Works regardless of how the model accesses the weight (module(x), F.linear, etc.)
import torch.nn.functional as F_torch


class LinearWithLoRA(nn.Module):
    """Drop-in replacement for nn.Linear that adds a LoRA low-rank update.
    The original weight is stored as a frozen buffer. LoRA A/B are trainable.
    Exposes .weight property so F.linear(x, module.weight) still works."""
    def __init__(self, original_linear, rank=32, alpha=32):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Store original weight as frozen buffer (not a parameter)
        self.register_buffer('frozen_weight', original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.register_buffer('frozen_bias', original_linear.bias.data.clone())
        else:
            self.frozen_bias = None

        # LoRA trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        """Returns frozen_weight + LoRA delta. Has grad_fn via lora_A/lora_B."""
        lora_A = self.lora_A.to(self.frozen_weight.dtype)
        lora_B = self.lora_B.to(self.frozen_weight.dtype)
        return self.frozen_weight + (lora_B @ lora_A) * self.scaling

    @property
    def bias(self):
        return self.frozen_bias

    def forward(self, x):
        return F_torch.linear(x, self.weight, self.frozen_bias)


def inject_lora_to_model(model, target_layers_start=0, rank=32, alpha=32, target_modules=['o_proj', 'v_proj']):
    # Freeze ALL parameters first
    logging.info("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Collect replacements (don't modify while iterating)
    replacements = []
    for name, module in model.named_modules():
        if 'layers' in name and isinstance(module, nn.Linear):
            parts = name.split('.')
            try:
                layers_idx = parts.index('layers')
                if layers_idx + 1 < len(parts):
                    layer_idx = int(parts[layers_idx + 1])
                else:
                    continue
            except (ValueError, IndexError):
                continue

            if layer_idx >= target_layers_start:
                module_name = parts[-1] if len(parts) > 0 else ''
                if module_name in target_modules:
                    replacements.append((name, parts, module))

    if len(replacements) == 0:
        logging.warning("No LoRA layers were injected! Check model structure and target_modules names.")
        logging.info("Available module names in model:")
        for name, module in model.named_modules():
            if 'layers' in name and isinstance(module, nn.Linear):
                logging.info(f"  {name}")
        return model, 0

    # Apply replacements
    for name, parts, original_module in replacements:
        lora_module = LinearWithLoRA(original_module, rank=rank, alpha=alpha)
        # Navigate to parent and replace the child module
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_module)
        logging.info(f"Replaced {name} with LinearWithLoRA (rank={rank}, in={original_module.in_features}, out={original_module.out_features})")

    logging.info(f"Total LoRA modules added: {len(replacements)}")
    return model, len(replacements)


class LoRAModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    def forward(self, input_ids, labels=None, use_cache=False, **kwargs):
        outputs = self.base_model(input_ids, labels=labels, use_cache=use_cache, **kwargs)
        return outputs

    def merge_lora_weights(self):
        """Merge LoRA into frozen weights and replace LinearWithLoRA back to nn.Linear."""
        logging.info("Merging LoRA weights into base model...")
        replacements = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, LinearWithLoRA):
                replacements.append((name, module))

        for name, module in replacements:
            # Create a standard nn.Linear with merged weights
            merged_linear = nn.Linear(
                module.in_features, module.out_features,
                bias=(module.frozen_bias is not None),
                dtype=module.frozen_weight.dtype,
                device=module.frozen_weight.device
            )
            with torch.no_grad():
                lora_A = module.lora_A.to(module.frozen_weight.dtype)
                lora_B = module.lora_B.to(module.frozen_weight.dtype)
                merged_linear.weight.copy_(module.frozen_weight + (lora_B @ lora_A) * module.scaling)
                if module.frozen_bias is not None:
                    merged_linear.bias.copy_(module.frozen_bias)

            # Replace in parent
            parts = name.split('.')
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], merged_linear)
            logging.info(f"Merged LoRA weights for {name}")

        logging.info("LoRA merge complete!")

    def save_pretrained(self, save_path, merge_lora=True, **kwargs):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if merge_lora:
            self.merge_lora_weights()
        self.base_model.save_pretrained(
            save_path,
            save_function=kwargs.get('save_function', torch.save)
        )
        logging.info(f"Saved model to {save_path}")


class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        # Qwen may not have bos_token_id, fall back to eos_token_id
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
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024,
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


def setup_logging(accelerator, project_name, args):
    logger = logging.getLogger(__name__)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_dir / f"debug_qwen_{accelerator.process_index}.log"),
            logging.StreamHandler()
        ]
    )

    if accelerator.is_main_process:
        if args.use_wandb:
            wandb.init(project=project_name, config=vars(args))
            run_name = wandb.run.name
        else:
            run_name = 'local_run'
        tb_writer = SummaryWriter(log_dir / "tensorboard_qwen")
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


def get_trainable_params(model, args):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            logging.info(f"Will optimize: {name}, shape: {param.shape}")

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found!")

    logging.info(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    return [{'params': trainable_params, 'weight_decay': args.weight_decay}]


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
    parser = argparse.ArgumentParser(description="Finetune Qwen2.5-Coder-7B with LoRA")

    # Model parameters
    parser.add_argument("--model_name", type=str, default=os.path.expanduser("~/models/Qwen2.5-Coder-7B-Instruct"), help="Base model path")

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target_layers_start", type=int, default=0, help="Starting layer for LoRA")
    parser.add_argument("--merge_lora_on_save", action="store_true", help="Merge LoRA weights when saving", default=True)
    parser.add_argument("--save_lora_only", action="store_true", help="Save only LoRA weights without merging")

    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--num_warmup_steps", type=int, default=3000, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_train_steps", type=int, default=12000, help="Maximum training steps")
    parser.add_argument("--max_eval_steps", type=int, default=0, help="Maximum evaluation steps")

    # Data parameters
    parser.add_argument("--train_dataset", type=str, default="../starcoder_20Btokens", help="Training dataset path")
    parser.add_argument("--valid_dataset", type=str, default="../datasets/starcoder_20Btokens_val", help="Validation dataset path")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--extrapolate_length", type=int, default=8192, help="Extrapolation length")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_checkpoint_steps", type=int, default=3000, help="Save checkpoint steps")
    parser.add_argument("--log_step", type=int, default=3000, help="Log steps")
    parser.add_argument("--output_dir", type=str, default="./results_qwen", help="Output directory")
    parser.add_argument("--project_name", type=str, default="COREGEN-Qwen", help="Project name")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    return parser.parse_args()


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

    logger, tb_writer, run_name = setup_logging(accelerator, args.project_name, args)
    logger.info(f"Accelerator state: {accelerator.state}")

    # Load Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load Qwen2.5-Coder-7B model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='eager',
    )

    # Qwen2.5-Coder-7B: 28 layers, attention uses q_proj/k_proj/v_proj/o_proj
    logger.info(f"Injecting LoRA (rank={args.lora_rank}) into layers >= {args.lora_target_layers_start}")
    base_model, lora_count = inject_lora_to_model(
        base_model,
        target_layers_start=args.lora_target_layers_start,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=['o_proj', 'v_proj', 'q_proj', 'k_proj']
    )

    model = LoRAModel(base_model)

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    train_dataloader, eval_dataloader, eval_extrapolation_dataloader = create_dataloaders(tokenizer, args)

    optimizer = AdamW(get_trainable_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    def get_lr():
        return optimizer.param_groups[0]['lr']

    model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Training loop
    model.train()
    completed_steps = 0

    # === DEBUG: single step diagnostics ===
    if accelerator.is_main_process:
        print("=" * 60)
        print(f"[DEBUG] torch.is_grad_enabled() = {torch.is_grad_enabled()}")
        print(f"[DEBUG] torch.is_inference_mode_enabled() = {torch.is_inference_mode_enabled()}")
        trainable = [(n, p.shape, p.requires_grad) for n, p in model.named_parameters() if p.requires_grad]
        print(f"[DEBUG] Trainable params count: {len(trainable)}")
        for n, s, rg in trainable[:5]:
            print(f"  {n}: shape={s}, requires_grad={rg}")
        # Check if LinearWithLoRA modules exist
        lora_modules = [(n, type(m).__name__) for n, m in model.named_modules() if isinstance(m, LinearWithLoRA)]
        print(f"[DEBUG] LinearWithLoRA modules: {len(lora_modules)}")
        if lora_modules:
            print(f"  first: {lora_modules[0][0]}")
        print("=" * 60)

    for step, batch in enumerate(tqdm(train_dataloader,
                                      total=args.max_train_steps,
                                      leave=False)):
        # DEBUG: check grad state before forward
        if step == 0 and accelerator.is_main_process:
            print(f"[DEBUG step=0] grad_enabled={torch.is_grad_enabled()}")
            print(f"[DEBUG step=0] inference_mode={torch.is_inference_mode_enabled()}")
            print(f"[DEBUG step=0] batch.requires_grad={batch.requires_grad if hasattr(batch, 'requires_grad') else 'N/A'}")

        outputs = model(batch, labels=batch, use_cache=False)

        # DEBUG: check outputs
        if step == 0 and accelerator.is_main_process:
            print(f"[DEBUG step=0] outputs.loss = {outputs.loss}")
            print(f"[DEBUG step=0] outputs.loss.requires_grad = {outputs.loss.requires_grad}")
            print(f"[DEBUG step=0] outputs.loss.grad_fn = {outputs.loss.grad_fn}")
            print(f"[DEBUG step=0] outputs.logits.requires_grad = {outputs.logits.requires_grad}")
            print(f"[DEBUG step=0] outputs.logits.grad_fn = {outputs.logits.grad_fn}")
            # Try computing loss externally
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = batch[..., 1:].contiguous().view(-1)
            ext_loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
            print(f"[DEBUG step=0] external_loss = {ext_loss}")
            print(f"[DEBUG step=0] external_loss.requires_grad = {ext_loss.requires_grad}")
            print(f"[DEBUG step=0] external_loss.grad_fn = {ext_loss.grad_fn}")

        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        print(loss)
        accelerator.backward(loss)

        if step % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        if step % args.save_checkpoint_steps == 0 and step > 0:
            logger.info('Evaluating and saving model checkpoint')

            metrics = {
                'lr': get_lr(),
                'samples': step * samples_per_step,
                'steps': completed_steps,
                'loss/train': loss.item(),
            }

            log_metrics(accelerator, logger, tb_writer, step, metrics, args.use_wandb)

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = output_dir / f"checkpoint_{step}"
            merge_lora = not args.save_lora_only
            unwrapped_model.save_pretrained(save_path, merge_lora=merge_lora, save_function=accelerator.save)

            # Also save tokenizer alongside checkpoint
            if accelerator.is_main_process:
                tokenizer.save_pretrained(save_path)

            model.train()

        if completed_steps >= args.max_train_steps:
            break

    # Save final checkpoint
    logger.info('Evaluating and saving final model checkpoint')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    final_save_path = output_dir / f"final_checkpoint_{step}"
    merge_lora = not args.save_lora_only
    unwrapped_model.save_pretrained(final_save_path, merge_lora=merge_lora, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_save_path)

    if accelerator.is_main_process and tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
