import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Callable, Union

import numpy as np
import pandas as pd
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
)


from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score, accuracy_score
from wandb import login


# -----------------------------
# Data
# -----------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

@dataclass(frozen=True)
class EncoderSpec:
    saved_model_name: str
    datamodule_cls: Type[Any]
    lit_model_cls: Type[Any]
    eval_fn: Callable[..., Dict[str, float]]
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"


def get_encoder_spec(version: int) -> EncoderSpec:
    specs: Dict[int, EncoderSpec] = {
        1: EncoderSpec(
            saved_model_name="saved_encoder_model_v1",
            datamodule_cls=EncoderDataModuleV1,
            lit_model_cls=EncoderLightningModuleV1,
            eval_fn=evaluate_best_v1,
        ),
        2: EncoderSpec(
            saved_model_name="saved_encoder_model_v2",
            datamodule_cls=EncoderDataModuleV2,
            lit_model_cls=EncoderLightningModuleV2,
            eval_fn=evaluate_best_v2,
        ),
    }
    if version not in specs:
        raise ValueError(f"Unknown VERSION={version}. Supported: {list(specs)}")
    return specs[version]


def resolve_best_ckpt(checkpoint_cb: ModelCheckpoint, ckpt_dir: str) -> str:
    best = checkpoint_cb.best_model_path

    # If eval_only, callback may not run — fall back to conventional names.
    if not best:
        for candidate in (
            os.path.join(ckpt_dir, "best.ckpt"),
            os.path.join(ckpt_dir, "best"),
            os.path.join(ckpt_dir, "last.ckpt"),
        ):
            if os.path.exists(candidate):
                best = candidate
                break

    if not best or not os.path.exists(best):
        raise FileNotFoundError(f"Could not find checkpoint in {ckpt_dir}")
    return best


# -------------------------
# Version 1 model (2 heads)
# -------------------------
class EncoderWithTwoHeads(torch.nn.Module):
    def __init__(self, model_name: str, num_labels1: int = 5, num_labels2: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier1 = torch.nn.Linear(hidden_size, num_labels1)
        self.classifier2 = torch.nn.Linear(hidden_size, num_labels2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits1 = self.classifier1(pooled)
        logits2 = self.classifier2(pooled)
        return logits1, logits2


# -------------------------
# DataModules
# -------------------------
class EncoderDataModuleV1(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        test_and_val_size: float,
        max_length: int,
        batch_size: int,
        num_workers: int,
        seed: int,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.test_and_val_size = test_and_val_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):

        # Load dataset and tokenizer
        load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.dataset = load_dataset(self.dataset_name)
        
        train_val_split = self.dataset["train"].train_test_split(test_size=self.test_and_val_size, seed=self.seed)
        train = train_val_split["train"]
        
        val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=self.seed)
        val = val_test_split["train"]
        test = val_test_split["test"]

        # label maps
        self.condition_map = {label: i for i, label in enumerate(set(train["condition"]))}
        self.reverse_condition_map = {v: k for k, v in self.condition_map.items()}
        self.record_type_map = {label: i for i, label in enumerate(set(train["record_type"]))}
        self.reverse_record_type_map = {v: k for k, v in self.record_type_map.items()}


        train = train.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        val = val.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        test = test.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        self.test = test  # keep raw for inference later
        self.train_ds = train
        self.val_ds = val
        self.test_ds = test

    def tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.tokenizer(
            f"Choose the condition and the record type of the following medical note: {example['text']}",
            truncation=True,
            max_length=self.max_length,
        )
        inputs["condition"] = self.condition_map[example["condition"]]
        inputs["record_type"] = self.record_type_map[example["record_type"]]
        return inputs

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=0,
        )
        condition = torch.tensor([ex["condition"] for ex in batch], dtype=torch.long)
        record_type = torch.tensor([ex["record_type"] for ex in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "condition": condition,
            "record_type": record_type,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )


class EncoderDataModuleV2(L.LightningDataModule):
    """
    MLM prompt formulation. We also compute max_tokens from label strings (condition + record_type).
    """
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        test_and_val_size: float,
        max_length: int,
        batch_size: int,
        num_workers: int,
        seed: int,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.test_and_val_size = test_and_val_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        # Load dataset and tokenizer
        load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.dataset = load_dataset(self.dataset_name)
        
        train_val_split = self.dataset["train"].train_test_split(test_size=self.test_and_val_size, seed=self.seed)
        train = train_val_split["train"]
        
        val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=self.seed)
        val = val_test_split["train"]
        test = val_test_split["test"]

        # label maps
        self.condition_map = {label: i for i, label in enumerate(set(train["condition"]))}
        self.reverse_condition_map = {v: k for k, v in self.condition_map.items()}
        self.record_type_map = {label: i for i, label in enumerate(set(train["record_type"]))}
        self.reverse_record_type_map = {v: k for k, v in self.record_type_map.items()}

        self.max_tokens = self._compute_max_tokens()

        train = train.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        val = val.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        test = test.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )

        self.test = test  # keep raw for inference later
        self.train_ds = train
        self.val_ds = val
        self.test_ds = test


    def _compute_max_tokens(self) -> int:
        max_tokens = 0
        for label in list(self.condition_map.keys()) + list(self.record_type_map.keys()):
            toks = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            max_tokens = max(max_tokens, len(toks))
        return max_tokens

    def tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        condition_tokens = self.tokenizer(example["condition"], add_special_tokens=False)["input_ids"]
        record_type_tokens = self.tokenizer(example["record_type"], add_special_tokens=False)["input_ids"]

        prompt = (
            f"Choose the condition and the record type of the following medical note: {example['text']}."
            f" The condition is: {' '.join([self.tokenizer.mask_token] * self.max_tokens)};"
            f" The record type is: {' '.join([self.tokenizer.mask_token] * self.max_tokens)}"
        )

        inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_length)

        token_ids = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        labels = [token if token == self.tokenizer.mask_token else -100 for token in token_ids]
        
        # Replace first set of [MASK] tokens with correct labels
        condition_done = False
        start_span = False
        start_span_idx = 0
        for i in range(len(labels)):
            if labels[i] == self.tokenizer.mask_token:
                if not start_span:
                    start_span = True
                    start_span_idx = i
            else:
                if start_span:
                    condition_done=True
                    start_span = False

            if start_span:
                if not condition_done:
                    labels[i] = condition_tokens[i - start_span_idx] if i-start_span_idx < len(condition_tokens) else self.tokenizer.pad_token_id
                else:
                    labels[i] = record_type_tokens[i - start_span_idx] if i-start_span_idx < len(record_type_tokens) else self.tokenizer.pad_token_id

        # Squeeze the tensors to (seq_len), since the tokenizer puts outputs the shape (1, seq_len)
        inputs["input_ids"] = torch.tensor(inputs["input_ids"], dtype=torch.long)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        inputs["condition"] = torch.tensor(self.condition_map[example["condition"]], dtype=torch.long)
        inputs["record_type"] = torch.tensor(self.record_type_map[example["record_type"]], dtype=torch.long)

        return inputs

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=-100,
        )
        condition = torch.stack([torch.tensor(ex["condition"]) for ex in batch])
        record_type = torch.stack([torch.tensor(ex["record_type"]) for ex in batch])


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "condition": condition,
            "record_type": record_type,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )


# -------------------------
# LightningModules
# -------------------------
class EncoderLightningModuleV1(L.LightningModule):
    def __init__(self, model_name: str, lr: float = 1e-5, num_labels1: int = 5, num_labels2: int = 2):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.save_hyperparameters()

        self.model = EncoderWithTwoHeads(model_name, num_labels1, num_labels2)
        self.loss_fn1 = torch.nn.CrossEntropyLoss()
        self.loss_fn2 = torch.nn.CrossEntropyLoss()

        # for epoch-end metrics
        self.val_preds1 = []
        self.val_preds2 = []
        self.val_true1 = []
        self.val_true2 = []

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        logits1, logits2 = self(batch["input_ids"], batch["attention_mask"])
        loss1 = self.loss_fn1(logits1, batch["condition"])
        loss2 = self.loss_fn2(logits2, batch["record_type"])
        loss = loss1 + loss2

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/loss1", loss1, on_step=True, on_epoch=False)
        self.log("train/loss2", loss2, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits1, logits2 = self(batch["input_ids"], batch["attention_mask"])
        loss1 = self.loss_fn1(logits1, batch["condition"])
        loss2 = self.loss_fn2(logits2, batch["record_type"])
        loss = loss1 + loss2

        pred1 = torch.argmax(logits1, dim=1)
        pred2 = torch.argmax(logits2, dim=1)

        self.val_preds1.append(pred1.detach().cpu())
        self.val_preds2.append(pred2.detach().cpu())
        self.val_true1.append(batch["condition"].detach().cpu())
        self.val_true2.append(batch["record_type"].detach().cpu())

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        p1 = torch.cat(self.val_preds1).numpy() if self.val_preds1 else np.array([])
        p2 = torch.cat(self.val_preds2).numpy() if self.val_preds2 else np.array([])
        t1 = torch.cat(self.val_true1).numpy() if self.val_true1 else np.array([])
        t2 = torch.cat(self.val_true2).numpy() if self.val_true2 else np.array([])

        self.val_preds1.clear()
        self.val_preds2.clear()
        self.val_true1.clear()
        self.val_true2.clear()

        if len(t1) > 0:
            f1_1 = f1_score(t1, p1, average="weighted")
            acc_1 = accuracy_score(t1, p1)
            self.log("val/f1_condition", f1_1, prog_bar=True)
            self.log("val/acc_condition", acc_1, prog_bar=True)

        if len(t2) > 0:
            f1_2 = f1_score(t2, p2, average="weighted")
            acc_2 = accuracy_score(t2, p2)
            self.log("val/f1_record_type", f1_2, prog_bar=True)
            self.log("val/acc_record_type", acc_2, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class EncoderLightningModuleV2(L.LightningModule):
    def __init__(self, model_name: str, lr: float = 1e-5):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.save_hyperparameters()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


# -------------------------
# Evaluation helpers (reload best ckpt easily)
# -------------------------
@torch.no_grad()
def evaluate_best_v1(
    ckpt_path: str,
    datamodule: EncoderDataModuleV1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    model = EncoderLightningModuleV1.load_from_checkpoint(ckpt_path, model_name=datamodule.tokenizer_name)
    model.to(device)
    model.eval()

    preds_cond, preds_rec = [], []
    true_cond, true_rec = [], []

    for batch in datamodule.test_ds:
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

        logits1, logits2 = model.model(input_ids=input_ids, attention_mask=attention_mask)
        c = torch.argmax(logits1, dim=1).item()
        r = torch.argmax(logits2, dim=1).item()

        preds_cond.append(c)
        preds_rec.append(r)

        true_cond.append(batch["condition"])
        true_rec.append(batch["record_type"])

    f1_cond = f1_score(true_cond, preds_cond, average="weighted")
    f1_rec = f1_score(true_rec, preds_rec, average="weighted")
    acc_cond = accuracy_score(true_cond, preds_cond)
    acc_rec = accuracy_score(true_rec, preds_rec)

    return {
        "test/f1_condition": f1_cond,
        "test/f1_record_type": f1_rec,
        "test/acc_condition": acc_cond,
        "test/acc_record_type": acc_rec,
    }


@torch.no_grad()
def evaluate_best_v2(
    ckpt_path: str,
    datamodule: EncoderDataModuleV2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    model = EncoderLightningModuleV2.load_from_checkpoint(ckpt_path, model_name=datamodule.tokenizer_name)
    model.to(device)
    model.eval()

    tokenizer = datamodule.tokenizer

    preds_cond, preds_rec = [], []
    true_cond, true_rec = [], []

    for batch in datamodule.test_ds:
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, T, V]

        mask_positions = torch.where(input_ids == tokenizer.mask_token_id)[1]
        pred_ids = input_ids.clone().squeeze(0)

        for pos in mask_positions:
            token_logits = logits[0, pos, :]
            pred_ids[pos] = torch.argmax(token_logits, dim=-1)

        decoded = tokenizer.decode(pred_ids, skip_special_tokens=True).lower()

        try:
            cond = decoded.split("the condition is:")[1].split(";")[0].strip()
            rec = decoded.split("the record type is:")[1].strip()
        except Exception:
            cond, rec = "", ""

        preds_cond.append(datamodule.condition_map.get(cond.title(), -1))
        preds_rec.append(datamodule.record_type_map.get(rec.title(), -1))
        true_cond.append(batch["condition"])
        true_rec.append(batch["record_type"])

    f1_cond = f1_score(true_cond, preds_cond, average="weighted")
    f1_rec = f1_score(true_rec, preds_rec, average="weighted")
    acc_cond = accuracy_score(true_cond, preds_cond)
    acc_rec = accuracy_score(true_rec, preds_rec)

    return {
        "test/f1_condition": f1_cond,
        "test/f1_record_type": f1_rec,
        "test/acc_condition": acc_cond,
        "test/acc_record_type": acc_rec,
    }


# -----------------------------
# Training loop (called by main)
# -----------------------------
def run_encoder_training(
    *,
    seed: int,
    version: int,
    model_name: str,
    num_epochs: int,
    grad_accum: int,
    early_stop_patience: int,
    max_length: int,
    batch_size: int,
    learning_rate: float,
    log_every_n_steps: int,
    save_dir: str,
    accelerator: str = "auto",
    devices: Union[str, int] = "auto",
    precision: str = "32-true",
    eval_only: bool = False,
    run_name: Optional[str] = None,
):
    L.seed_everything(seed, workers=True)

    spec = get_encoder_spec(version)

    run_name = spec.saved_model_name
    ckpt_dir = os.path.join(save_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- DataModule / Model ----
    dm = spec.datamodule_cls(
        dataset_name="karenwky/pet-health-symptoms-dataset",
        tokenizer_name=model_name,
        test_and_val_size=0.2,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=4,
        seed=seed,
    )
    lit_model = spec.lit_model_cls(model_name=model_name, lr=learning_rate)


    # ---- Logger ----
    wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        # Authenticate using the key
        login(key=wandb_key)
        print("Successfully logged into wandb.")
    else:
        print("Error: WANDB_API_KEY environment variable not found.")
    logger = WandbLogger(
        project="oceanprotocol",
        name=run_name,
    )

    # ---- Checkpointing / Early stopping ----
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        save_top_k=1,
        monitor=spec.monitor_metric,
        mode=spec.monitor_mode,
        save_last=False,  # set to True if you also want to keep track of the last epoch's checkpoint (handy for resuming if interrupted, but can use more storage
    )
    earlystop_cb = EarlyStopping(
        monitor=spec.monitor_metric,
        mode=spec.monitor_mode,
        patience=early_stop_patience,
    )

    # ---- Trainer ----
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=num_epochs,
        accumulate_grad_batches=grad_accum,
        callbacks=[checkpoint_cb, earlystop_cb],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=ckpt_dir,
        deterministic=True,
    )

    dm.setup()

    if not eval_only:
        trainer.fit(lit_model, datamodule=dm)

    # ---- Resolve best checkpoint, save tokenizer, eval ----
    best_ckpt_path = resolve_best_ckpt(checkpoint_cb, ckpt_dir)

    metrics = spec.eval_fn(best_ckpt_path, dm)
    print("\n=== Best checkpoint test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nBest checkpoint: {best_ckpt_path}")
    print(f"Tokenizer saved to: {ckpt_dir}")


# -------------------------
# Main
# -------------------------
def main():
    # -------------------------
    # Constants (all here)
    # -------------------------
    SEED = 42

    # Model
    MODEL_NAME = "answerdotai/ModernBERT-large"

    # Tokenization / batching
    MAX_LENGTH = 1028
    BATCH_SIZE = 16              # per GPU
    GRAD_ACCUM = 1                   # effective batch = BATCH_SIZE * GRAD_ACCUM

    # Training
    LR = 5e-5
    LOG_EVERY_N_STEPS = 1
    NUM_EPOCHS = 30
    EARLY_STOP_PATIENCE = 5

    # Precision
    PRECISION = "bf16-mixed"  # if your GPUs support bf16; otherwise use "16-mixed"

    # Output
    SAVE_DIR = "/data/outputs/encoder_finetuning"

    # Version: 
    # 1: Encoder backbone with classification heads
    # 2: Encoder with masked token prediction-style classification
    VERSION = 2

    wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        # Authenticate using the key
        login(key=wandb_key)
        print("Successfully logged into wandb.")
    else:
        print("Error: WANDB_API_KEY environment variable not found.")

    run_encoder_training(
        seed=SEED,
        version=VERSION,
        model_name=MODEL_NAME,
        num_epochs=NUM_EPOCHS,
        grad_accum=GRAD_ACCUM,
        early_stop_patience=EARLY_STOP_PATIENCE,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        save_dir=SAVE_DIR,
        accelerator="auto",
        devices="auto",
        precision=PRECISION,
        eval_only=False,  # True to skip training and only eval best ckpt
        run_name=None,  # or "my_run_name"
    )


if __name__ == "__main__":
    main()
