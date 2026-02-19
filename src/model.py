"""
NER model training, evaluation, and management.

Wraps HuggingFace Transformers for fine-tuning pretrained BERT/RoBERTa
models on token classification tasks with seqeval-based evaluation.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class NERModel:
    """Named Entity Recognition model backed by a pretrained transformer.

    Supports BERT, RoBERTa, and other HuggingFace token classification
    models. Handles fine-tuning, evaluation with seqeval, inference,
    and model export.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        label_list: Optional[List[str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.label_list = label_list or [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
            "B-LOC", "I-LOC", "B-MISC", "I-MISC",
        ]
        self.label2id = label2id or {l: i for i, l in enumerate(self.label_list)}
        self.id2label = id2label or {i: l for i, l in enumerate(self.label_list)}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load a pretrained model with a token classification head.

        Args:
            model_path: Path to a fine-tuned model. If None, loads the base model.
        """
        path = model_path or self.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            path,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        logger.info(f"Model loaded from {path} with {len(self.label_list)} labels")

    def setup_training(
        self,
        output_dir: str = "./models/ner_model",
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        fp16: bool = False,
    ) -> TrainingArguments:
        """Configure training arguments and data collator.

        Args:
            output_dir: Directory for model checkpoints and logs.
            num_epochs: Number of training epochs.
            batch_size: Batch size for training and evaluation.
            learning_rate: Learning rate for AdamW optimizer.
            weight_decay: L2 regularization coefficient.
            warmup_steps: Linear warmup steps for learning rate scheduler.
            save_strategy: When to save checkpoints ("epoch", "steps", "no").
            evaluation_strategy: When to evaluate ("epoch", "steps", "no").
            fp16: Whether to use mixed precision training.

        Returns:
            Configured TrainingArguments object.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            logging_steps=50,
            fp16=fp16 and torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=True,
        )
        return training_args

    def _compute_metrics(self, eval_preds: Tuple) -> Dict[str, float]:
        """Compute seqeval metrics from model predictions.

        Converts token-level predictions back to entity-level BIO tags
        and computes precision, recall, and F1 using seqeval.

        Args:
            eval_preds: Tuple of (predictions logits, label_ids).

        Returns:
            Dictionary with precision, recall, f1, and accuracy.
        """
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            pred_tags = []
            true_tags = []

            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue

                pred_tag = self.id2label.get(pred_id, "O")
                true_tag = self.id2label.get(label_id, "O")
                pred_tags.append(pred_tag)
                true_tags.append(true_tag)

            true_predictions.append(pred_tags)
            true_labels.append(true_tags)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    def train(
        self,
        tokenized_datasets: DatasetDict,
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        output_dir: str = "./models/ner_model",
        patience: int = 3,
    ) -> Dict[str, Any]:
        """Fine-tune the model using HuggingFace Trainer.

        Args:
            tokenized_datasets: DatasetDict with train, validation, test splits.
            num_epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            output_dir: Output directory.
            patience: Early stopping patience in evaluation rounds.

        Returns:
            Training results dictionary with metrics.
        """
        if self.model is None:
            self.load_model()

        training_args = self.setup_training(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        logger.info("Starting training...")
        train_result = self.trainer.train()

        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metrics = train_result.metrics
        logger.info(f"Training completed: {metrics}")

        return {
            "train_metrics": metrics,
            "model_path": output_dir,
        }

    def evaluate(
        self, tokenized_datasets: Optional[DatasetDict] = None, split: str = "test"
    ) -> Dict[str, Any]:
        """Evaluate model and return seqeval metrics per entity type.

        Args:
            tokenized_datasets: DatasetDict to evaluate on. Uses trainer's eval dataset if None.
            split: Which split to evaluate ("test", "validation").

        Returns:
            Dictionary with per-entity and overall metrics.
        """
        if self.trainer is None:
            raise RuntimeError("Model must be trained before evaluation.")

        eval_dataset = tokenized_datasets[split] if tokenized_datasets else None
        results = self.trainer.evaluate(eval_dataset=eval_dataset)

        predictions, labels, _ = self.trainer.predict(
            eval_dataset or self.trainer.eval_dataset
        )
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            pred_tags = []
            true_tags = []
            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                pred_tags.append(self.id2label.get(pred_id, "O"))
                true_tags.append(self.id2label.get(label_id, "O"))
            true_predictions.append(pred_tags)
            true_labels.append(true_tags)

        report = classification_report(true_labels, true_predictions, output_dict=True)
        report_str = classification_report(true_labels, true_predictions)

        logger.info(f"\n{report_str}")

        return {
            "eval_metrics": results,
            "classification_report": report,
            "classification_report_str": report_str,
        }

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Run inference on raw text.

        Args:
            text: Input text string.

        Returns:
            List of token-level predictions with labels and scores.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before prediction.")

        self.model.eval()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        confidences = probabilities.max(dim=-1).values[0].cpu().tolist()

        results = []
        for idx, (pred_id, conf, offset) in enumerate(
            zip(predictions, confidences, offset_mapping)
        ):
            if offset == [0, 0]:
                continue

            label = self.id2label.get(pred_id, "O")
            token_text = text[offset[0]:offset[1]]

            results.append({
                "token": token_text,
                "label": label,
                "confidence": round(conf, 4),
                "start": offset[0],
                "end": offset[1],
            })

        return results

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Post-process predictions into entity spans.

        Merges consecutive BIO-tagged tokens into complete entity mentions.

        Args:
            text: Input text.

        Returns:
            List of entities with text, label, start, end, confidence.
        """
        token_predictions = self.predict(text)

        entities = []
        current_entity = None

        for pred in token_predictions:
            label = pred["label"]

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "text": pred["token"],
                    "label": entity_type,
                    "start": pred["start"],
                    "end": pred["end"],
                    "confidence": pred["confidence"],
                    "token_count": 1,
                }

            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    gap = text[current_entity["end"]:pred["start"]]
                    current_entity["text"] += gap + pred["token"]
                    current_entity["end"] = pred["end"]
                    current_entity["confidence"] = min(
                        current_entity["confidence"], pred["confidence"]
                    )
                    current_entity["token_count"] += 1
                else:
                    entities.append(current_entity)
                    current_entity = None

            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def save_model(self, output_dir: str) -> None:
        """Save model and tokenizer to disk.

        Args:
            output_dir: Directory to save the model.
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.model:
            self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def load_from_path(self, model_path: str) -> None:
        """Load a fine-tuned model from a directory.

        Args:
            model_path: Path containing model files.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.label_list = [self.id2label[i] for i in sorted(self.id2label.keys())]

        logger.info(f"Loaded fine-tuned model from {model_path}")

    def export_onnx(self, output_path: str, opset_version: int = 14) -> str:
        """Export the model to ONNX format for optimized inference.

        Args:
            output_path: File path for the ONNX model.
            opset_version: ONNX opset version.

        Returns:
            Path to the exported ONNX model.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before export.")

        self.model.eval()
        self.model.to("cpu")

        dummy_input = self.tokenizer(
            "This is a sample sentence for export.",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            opset_version=opset_version,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
        )

        self.model.to(self.device)
        logger.info(f"Model exported to ONNX at {output_path}")
        return output_path
