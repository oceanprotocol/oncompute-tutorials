import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from lightning.pytorch.callbacks import TQDMProgressBar

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from wandb import login


# -----------------------------
# Data
# -----------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SlimOrcaDataModule(L.LightningDataModule):
    """
    Loads Open-Orca/SlimOrca-Dedup (ShareGPT-style `conversations`) and prepares
    batches for SFT on the *last assistant turn*.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        train_split: str,
        val_split: str,
        val_size: int,
        max_length: int,
        micro_batch_size: int,
        num_workers: int,
        seed: int,
        enable_thinking: bool,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.train_split = train_split
        self.val_split = val_split
        self.val_size = val_size
        self.max_length = max_length
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.enable_thinking = enable_thinking

        self.tokenizer = None
        self._setup_done = False

    def prepare_data(self):
        # Runs only on global rank 0
        load_dataset(self.dataset_name)

    def _mark_good(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        msgs = self._extract_messages(ex)
        ptxt, ftxt = self._build_prompt_and_full_text(msgs)

        # Matches the old: if not ptxt or not ftxt -> skip
        if not ptxt or not ftxt:
            return {"prompt_text": "", "full_text": "", "is_good": False}

        # Matches the old "all labels masked" case, including truncation effects.
        # We check whether (after truncation) there is at least 1 token beyond prompt.
        full_ids = self.tokenizer(
            ftxt, truncation=True, max_length=self.max_length, add_special_tokens=True
        )["input_ids"]
        prompt_ids = self.tokenizer(
            ptxt, truncation=True, max_length=self.max_length, add_special_tokens=True
        )["input_ids"]

        # In collate we mask [:prompt_len]; if prompt_len >= full_len => nothing left to learn.
        is_good = len(full_ids) > len(prompt_ids)

        return {"prompt_text": ptxt, "full_text": ftxt, "is_good": is_good}
    
    def _prep_and_filter(self, split_ds):
        split_ds = split_ds.map(self._mark_good, desc="Building prompt/full + validity")
        split_ds = split_ds.filter(lambda ex: ex["is_good"], desc="Filtering bad examples")
        # Keep prompt/full for collate; drop the flag
        return split_ds.remove_columns(["is_good"])

    def setup(self, stage: Optional[str] = None):
        if not self._setup_done:
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                padding_side="left",
                use_fast=True,
                trust_remote_code=True,
            )
            # Make sure we have a pad token (Qwen tokenizers typically do, but be safe)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Dataset
            ds = load_dataset(self.dataset_name)

            if self.train_split not in ds:
                raise ValueError(f"Split '{self.train_split}' not in dataset splits: {list(ds.keys())}")
            

            print("Filtering out problematic samples for training", flush=True)
            train_full = self._prep_and_filter(ds[self.train_split])

            # If val split exists, filter it too; otherwise create val/test from filtered train_full.
            if self.val_split in ds:
                print("Filtering out problematic samples for validation", flush=True)
                val = self._prep_and_filter(ds[self.val_split])

                # Create test split same size as val (from dataset test split if exists, else from train_full).
                val_n = len(val)

                if "test" in ds:
                    print("Filtering out problematic samples for testing", flush=True)
                    test_full = self._prep_and_filter(ds["test"])
                    test_n = min(val_n, len(test_full))
                    test = test_full.shuffle(seed=self.seed).select(range(test_n))
                else:
                    # sample from train_full without overlap with train by taking from a shuffled view
                    shuffled = train_full.shuffle(seed=self.seed)
                    test_n = min(val_n, len(shuffled))
                    test = shuffled.select(range(test_n))

                train = train_full

            else:
                # deterministic split: val then test then train
                shuffled = train_full.shuffle(seed=self.seed)

                val_n = min(self.val_size, len(shuffled))
                test_n = min(val_n, max(0, len(shuffled) - val_n))  # same size as val if possible

                val = shuffled.select(range(val_n))
                test = shuffled.select(range(val_n, val_n + test_n))
                train = shuffled.select(range(val_n + test_n, len(shuffled)))

            print("Datasets are ready", flush=True)
            self.train_ds = train
            self.val_ds = val
            self.test_ds = test
            self._setup_done = True

    def _extract_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        SlimOrca-Dedup uses a ShareGPT-like schema:
        example["conversations"] = [{"from": "system"/"human"/"gpt", "value": "..."}...]
        """
        conv = example.get("conversations", None)
        if conv is None:
            raise KeyError("Expected a 'conversations' field in the dataset example.")

        role_map = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
        }

        messages: List[Dict[str, str]] = []
        for turn in conv:
            frm = turn.get("from")
            val = turn.get("value")
            if frm not in role_map or val is None:
                continue
            messages.append({"role": role_map[frm], "content": val})
        return messages

    def _build_prompt_and_full_text(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        We train only on the last assistant message.

        prompt_messages = messages[:-1] + generation prompt
        full_messages = messages (includes last assistant message)

        If the last message isn't assistant, we skip by returning ("","").
        """
        if len(messages) < 2:
            return "", ""

        if messages[-1]["role"] != "assistant":
            return "", ""

        prompt_messages = messages[:-1]
        full_messages = messages

        # Qwen3 supports apply_chat_template; enable_thinking default can inject <think> blocks.
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        return prompt_text, full_text

    def _collate(self, examples: List[Dict[str, Any]]) -> Batch:
        # examples are guaranteed "good" by setup(); prompt/full already computed
        prompt_texts = [ex["prompt_text"] for ex in examples]
        full_texts   = [ex["full_text"] for ex in examples]

        full_enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        labels = input_ids.clone()

        prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()
        for i, plen in enumerate(prompt_lens):
            labels[i, :int(plen)] = -100

        labels[attention_mask == 0] = -100

        return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
            drop_last=False,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
            drop_last=False,
        )


# -----------------------------
# Model
# -----------------------------
class SFTLoRAModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
        grad_clip: float,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: List[str],
        save_dir: str,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # PEFT LoRA
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)

        self.max_steps = max_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        # Helpful: prints trainable params on rank 0 only
        self._log_trainable_params()

    @rank_zero_only
    def _log_trainable_params(self):
        trainable = 0
        total = 0
        for _, p in self.model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"[PEFT] Trainable params: {trainable:,} / {total:,} "
              f"({100.0 * trainable / total:.4f}%)")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: Batch, batch_idx: int):
        out = self(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        loss = out.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch.input_ids.size(0), sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        out = self(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        loss = out.loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.input_ids.size(0), sync_dist=True)
        return loss
    
    def test_step(self, batch: Batch, batch_idx: int):
        out = self(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        loss = out.loss
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.input_ids.size(0), sync_dist=True)
        return loss

    def configure_optimizers(self):
        # AdamW
        optimizer = AdamW8bit(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Linear warmup then linear decay
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
class OptimStepProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            bar.total = self.trainer.max_steps
        return bar

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Show optimizer-step progress, not batch progress
        if self.train_progress_bar is not None:
            self.train_progress_bar.n = trainer.global_step
            self.train_progress_bar.set_postfix({
                "step": f"{trainer.global_step}/{trainer.max_steps}"
            })
            self.train_progress_bar.refresh()

# -----------------------------
# Training loop (called by main)
# -----------------------------
def run_training(
    *,
    seed: int,
    dataset_name: str,
    model_name: str,
    max_length: int,
    micro_batch_size: int,
    grad_accum: int,
    num_workers: int,
    val_size: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
    log_every_n_steps: int,
    val_check_interval: int,
    save_dir: str,
    precision: str,
    devices: int,
    strategy: str,
    enable_thinking: bool,
    # LoRA
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
):
    L.seed_everything(seed, workers=True)


    dm = SlimOrcaDataModule(
        dataset_name=dataset_name,
        tokenizer_name=model_name,
        train_split="train",
        val_split="validation",  # if missing, we create one from train
        val_size=val_size,
        max_length=max_length,
        micro_batch_size=micro_batch_size,
        num_workers=num_workers,
        seed=seed,
        enable_thinking=enable_thinking,
    )

    lit_model = SFTLoRAModule(
        model_name=model_name,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        grad_clip=1.0,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        save_dir=save_dir,
    )

    # ---- W&B Logger ----
    wandb_logger = WandbLogger(
        project="oceanprotocol",
        name="qwen3-8b-slimorca",
    )

    # ---- Checkpointing ----
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="step-{step}-val_loss-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices if torch.cuda.is_available() else 1,
        strategy=strategy if torch.cuda.is_available() and devices > 1 else "auto",
        precision=precision if torch.cuda.is_available() else "32-true",
        max_steps=max_steps,
        accumulate_grad_batches=grad_accum,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, OptimStepProgressBar()],
        default_root_dir=save_dir,
        deterministic=True
    )

    from pathlib import Path
    path = Path("/data/outputs/my_folder/test.txt") 
    path.parent.mkdir(parents=True, exist_ok=True) 
    with path.open("w", encoding="utf-8") as f: 
        f.write("test works")

    print("Starting Initial Validation!", flush=True)
    trainer.validate(lit_model, datamodule=dm)
    print("Starting Training!", flush=True)
    trainer.fit(lit_model, datamodule=dm)
    print("Starting Final Testing!", flush=True)
    trainer.test(model=lit_model, datamodule=dm)


# -----------------------------
# main
# -----------------------------
def main():
    # -------------------------
    # Constants (all here)
    # -------------------------
    SEED = 42

    # Dataset
    DATASET_NAME = "Open-Orca/SlimOrca-Dedup"
    VAL_SIZE = 2000  # created from train if no validation split exists

    # Model
    MODEL_NAME = "Qwen/Qwen3-8B"

    # Qwen3 chat template option: disable thinking so training text doesn't include <think> blocks.
    ENABLE_THINKING = False

    # Tokenization / batching
    MAX_LENGTH = 4096
    MICRO_BATCH_SIZE = 1              # per GPU
    GRAD_ACCUM = 16                   # effective batch = MICRO_BATCH_SIZE * GRAD_ACCUM * num_gpus
    NUM_WORKERS = 4

    # Training
    LR = 1e-6
    WEIGHT_DECAY = 0.0
    WARMUP_STEPS = 200
    MAX_STEPS = 1000
    LOG_EVERY_N_STEPS = 1
    VAL_CHECK_INTERVAL = 500

    # Multi-GPU
    DEVICES = torch.cuda.device_count()   # set NUM_GPUS=8 etc.
    STRATEGY = "ddp_find_unused_parameters_false"    # often best for HF + PEFT

    # Precision
    PRECISION = "bf16-mixed"  # if your GPUs support bf16; otherwise use "16-mixed"

    # Output
    SAVE_DIR = "/data/outputs/qwen3_8b_slimorca_lora_adapter"

    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # Target modules:
    # Qwen-style architectures commonly respond well to targeting attention + MLP projections.
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        # Authenticate using the key
        login(key=wandb_key)
        print("Successfully logged into wandb.")
    else:
        print("Error: WANDB_API_KEY environment variable not found.")

    # -------------------------
    # Call training loop
    # -------------------------
    run_training(
        seed=SEED,
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
        num_workers=NUM_WORKERS,
        val_size=VAL_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        val_check_interval=VAL_CHECK_INTERVAL,
        save_dir=SAVE_DIR,
        precision=PRECISION,
        devices=DEVICES,
        strategy=STRATEGY,
        enable_thinking=ENABLE_THINKING,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
    )


if __name__ == "__main__":
    main()
