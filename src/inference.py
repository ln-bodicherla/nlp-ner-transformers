"""
NER inference pipeline.

Provides a streamlined interface for entity extraction from text,
including sub-word merging, batch processing, and formatted output.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

logger = logging.getLogger(__name__)


class NERInference:
    """Production inference wrapper for trained NER models.

    Handles model loading, single and batch prediction, sub-word token
    merging, and output formatting.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        aggregation_strategy: str = "simple",
    ):
        self.model_path = model_path
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 0 if self.device_str == "cuda" else -1
        self.aggregation_strategy = aggregation_strategy

        self.model = None
        self.tokenizer = None
        self.pipe: Optional[TokenClassificationPipeline] = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

        self.load_pipeline()

    def load_pipeline(self) -> None:
        """Load trained model as a HuggingFace pipeline for inference.

        Initializes the tokenizer, model, and token classification pipeline
        with the specified aggregation strategy.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        self.pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            aggregation_strategy=self.aggregation_strategy,
        )

        logger.info(
            f"Loaded NER pipeline from {self.model_path} "
            f"with {len(self.id2label)} labels on {self.device_str}"
        )

    def predict_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from a single text input.

        Args:
            text: Input text string.

        Returns:
            List of entity dictionaries with text, label, start, end, confidence.
        """
        if not text or not text.strip():
            return []

        raw_entities = self.pipe(text)
        entities = self._post_process_entities(raw_entities, text)
        return entities

    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """Run batch inference on multiple texts.

        Args:
            texts: List of input text strings.
            batch_size: Number of texts per batch.

        Returns:
            List of entity lists, one per input text.
        """
        if not texts:
            return []

        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_entities = self.pipe(batch)

            if isinstance(batch_entities[0], dict):
                batch_entities = [batch_entities]

            for text, raw_entities in zip(batch, batch_entities):
                entities = self._post_process_entities(raw_entities, text)
                all_results.append(entities)

        return all_results

    def _post_process_entities(
        self, raw_entities: List[Dict], text: str
    ) -> List[Dict[str, Any]]:
        """Clean and format raw pipeline output.

        Handles score rounding, entity text extraction, and filtering of
        low-confidence predictions.

        Args:
            raw_entities: Raw output from the HuggingFace pipeline.
            text: Original input text for span extraction.

        Returns:
            Cleaned list of entity dictionaries.
        """
        entities = []

        for entity in raw_entities:
            label = entity.get("entity_group", entity.get("entity", "O"))
            if label == "O":
                continue

            label = label.replace("B-", "").replace("I-", "")

            score = entity.get("score", 0.0)
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            word = entity.get("word", text[start:end])

            word = word.strip()
            if word.startswith("##"):
                word = word[2:]

            entities.append({
                "text": word,
                "label": label,
                "start": start,
                "end": end,
                "confidence": round(float(score), 4),
            })

        entities = self.merge_subword_entities(entities, text)
        return entities

    def merge_subword_entities(
        self, entities: List[Dict[str, Any]], text: str
    ) -> List[Dict[str, Any]]:
        """Merge sub-word tokens into complete entity spans.

        When multiple consecutive sub-word tokens share the same entity label
        and are adjacent in the text, they are combined into a single entity.

        Args:
            entities: List of per-token entity predictions.
            text: Original text for accurate span extraction.

        Returns:
            List of merged entities.
        """
        if not entities:
            return []

        merged = []
        current = None

        for entity in entities:
            if current is None:
                current = dict(entity)
                continue

            same_label = entity["label"] == current["label"]
            adjacent = entity["start"] - current["end"] <= 1

            if same_label and adjacent:
                current["end"] = entity["end"]
                current["text"] = text[current["start"]:current["end"]].strip()
                current["confidence"] = min(current["confidence"], entity["confidence"])
            else:
                merged.append(current)
                current = dict(entity)

        if current is not None:
            merged.append(current)

        return merged

    def format_output(
        self,
        entities: List[Dict[str, Any]],
        text: str,
        output_format: str = "dict",
    ) -> Any:
        """Format entity predictions into structured output.

        Args:
            entities: List of extracted entities.
            text: Original input text.
            output_format: "dict", "html", or "terminal".

        Returns:
            Formatted output in the specified format.
        """
        if output_format == "dict":
            return {
                "text": text,
                "entities": entities,
                "entity_count": len(entities),
                "labels_found": list(set(e["label"] for e in entities)),
            }

        elif output_format == "html":
            return self._format_html(entities, text)

        elif output_format == "terminal":
            return self._format_terminal(entities, text)

        else:
            raise ValueError(f"Unknown format: {output_format}. Use 'dict', 'html', or 'terminal'.")

    def _format_html(self, entities: List[Dict[str, Any]], text: str) -> str:
        """Generate HTML with highlighted entities.

        Args:
            entities: Extracted entities.
            text: Original text.

        Returns:
            HTML string with color-coded entity spans.
        """
        color_map = {
            "PER": "#FBBF24",
            "ORG": "#60A5FA",
            "LOC": "#34D399",
            "MISC": "#F472B6",
        }

        sorted_entities = sorted(entities, key=lambda e: e["start"])
        html_parts = []
        last_end = 0

        for entity in sorted_entities:
            if entity["start"] > last_end:
                html_parts.append(text[last_end:entity["start"]])

            color = color_map.get(entity["label"], "#D1D5DB")
            html_parts.append(
                f'<mark style="background-color: {color}; padding: 2px 4px; '
                f'border-radius: 3px;" title="{entity["label"]} '
                f'({entity["confidence"]:.2f})">'
                f'{entity["text"]}'
                f'<sup style="font-size: 0.7em;">{entity["label"]}</sup>'
                f'</mark>'
            )
            last_end = entity["end"]

        if last_end < len(text):
            html_parts.append(text[last_end:])

        return "".join(html_parts)

    def _format_terminal(self, entities: List[Dict[str, Any]], text: str) -> str:
        """Generate terminal-colored output with entity highlights.

        Args:
            entities: Extracted entities.
            text: Original text.

        Returns:
            ANSI-colored string for terminal display.
        """
        color_codes = {
            "PER": "\033[93m",
            "ORG": "\033[94m",
            "LOC": "\033[92m",
            "MISC": "\033[95m",
        }
        reset = "\033[0m"
        bold = "\033[1m"

        sorted_entities = sorted(entities, key=lambda e: e["start"])
        parts = []
        last_end = 0

        for entity in sorted_entities:
            if entity["start"] > last_end:
                parts.append(text[last_end:entity["start"]])

            color = color_codes.get(entity["label"], "\033[97m")
            parts.append(
                f"{bold}{color}[{entity['text']}]({entity['label']}){reset}"
            )
            last_end = entity["end"]

        if last_end < len(text):
            parts.append(text[last_end:])

        return "".join(parts)

    def highlight_entities(
        self, text: str, output_format: str = "terminal"
    ) -> str:
        """Extract entities and return highlighted text.

        Convenience method that combines prediction and formatting.

        Args:
            text: Input text.
            output_format: "html" or "terminal".

        Returns:
            Highlighted text string.
        """
        entities = self.predict_text(text)
        return self.format_output(entities, text, output_format=output_format)

    def get_supported_labels(self) -> List[str]:
        """Return list of entity labels the model can predict.

        Returns:
            Sorted list of unique entity type labels (without BIO prefixes).
        """
        labels = set()
        for label in self.id2label.values():
            if label == "O":
                continue
            clean = label.replace("B-", "").replace("I-", "")
            labels.add(clean)
        return sorted(labels)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration.

        Returns:
            Dictionary with model path, device, labels, and config details.
        """
        config = self.model.config if self.model else {}

        return {
            "model_path": self.model_path,
            "device": self.device_str,
            "num_labels": len(self.id2label),
            "supported_labels": self.get_supported_labels(),
            "model_type": getattr(config, "model_type", "unknown"),
            "hidden_size": getattr(config, "hidden_size", 0),
            "num_attention_heads": getattr(config, "num_attention_heads", 0),
            "aggregation_strategy": self.aggregation_strategy,
        }
