"""
NER data processing pipeline.

Handles loading CoNLL and JSON annotated data, building label mappings,
sub-word tokenization with label alignment, and dataset preparation
for HuggingFace Trainer.
"""

import json
import logging
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class NERDataProcessor:
    """Processes and prepares NER datasets for transformer-based training.

    Handles multiple input formats (CoNLL, JSON), builds label mappings,
    performs sub-word tokenization with correct BIO label alignment, and
    outputs HuggingFace Dataset objects.
    """

    def __init__(
        self,
        label_list: Optional[List[str]] = None,
        max_length: int = 512,
    ):
        self.label_list = label_list or [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
            "B-LOC", "I-LOC", "B-MISC", "I-MISC",
        ]
        self.max_length = max_length
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None

        self.create_label_mapping()

    def create_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build label-to-id and id-to-label dictionaries.

        Returns:
            Tuple of (label2id, id2label) dictionaries.
        """
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
        return self.label2id, self.id2label

    def load_conll_format(self, file_path: str) -> Dataset:
        """Read CoNLL-2003 style NER data.

        Expected format: one token per line with tab/space-separated fields.
        Sentences separated by blank lines.
        Format: word [POS] [chunk] NER_tag

        Args:
            file_path: Path to the CoNLL format file.

        Returns:
            HuggingFace Dataset with 'tokens' and 'ner_tags' columns.
        """
        sentences = []
        tags = []
        current_tokens = []
        current_tags = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line.startswith("-DOCSTART-") or line == "":
                    if current_tokens:
                        sentences.append(current_tokens)
                        tags.append(current_tags)
                        current_tokens = []
                        current_tags = []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[-1]

                    if tag not in self.label2id:
                        tag = "O"

                    current_tokens.append(token)
                    current_tags.append(self.label2id[tag])

            if current_tokens:
                sentences.append(current_tokens)
                tags.append(current_tags)

        dataset = Dataset.from_dict({
            "tokens": sentences,
            "ner_tags": tags,
        })

        logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        return dataset

    def load_json_format(self, file_path: str) -> Dataset:
        """Read JSON annotated data in Label Studio export format.

        Expected JSON structure:
        [
            {
                "text": "full sentence text",
                "annotations": [
                    {"start": 0, "end": 5, "label": "PER", "text": "token"}
                ]
            }
        ]

        Args:
            file_path: Path to the JSON file.

        Returns:
            HuggingFace Dataset with 'tokens' and 'ner_tags' columns.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_tokens = []
        all_tags = []

        for entry in data:
            text = entry["text"]
            annotations = sorted(entry.get("annotations", []), key=lambda x: x["start"])

            tokens = text.split()
            tags = [self.label2id["O"]] * len(tokens)

            char_to_token = {}
            current_pos = 0
            for tidx, token in enumerate(tokens):
                start = text.find(token, current_pos)
                for char_pos in range(start, start + len(token)):
                    char_to_token[char_pos] = tidx
                current_pos = start + len(token)

            for ann in annotations:
                ann_start = ann["start"]
                ann_end = ann["end"]
                label = ann["label"]

                b_tag = f"B-{label}"
                i_tag = f"I-{label}"

                first_token = True
                for char_pos in range(ann_start, ann_end):
                    if char_pos in char_to_token:
                        tidx = char_to_token[char_pos]
                        if first_token:
                            if b_tag in self.label2id:
                                tags[tidx] = self.label2id[b_tag]
                            first_token = False
                        else:
                            if i_tag in self.label2id:
                                tags[tidx] = self.label2id[i_tag]

            all_tokens.append(tokens)
            all_tags.append(tags)

        dataset = Dataset.from_dict({
            "tokens": all_tokens,
            "ner_tags": all_tags,
        })

        logger.info(f"Loaded {len(data)} annotations from {file_path}")
        return dataset

    def tokenize_and_align_labels(
        self, examples: Dict[str, List], tokenizer: PreTrainedTokenizerFast
    ) -> Dict[str, Any]:
        """Handle sub-word tokenization with BIO label alignment.

        When a word is split into sub-word tokens, the first sub-token
        inherits the word's label, and subsequent sub-tokens get -100
        (ignored in loss computation). Special tokens also get -100.

        Args:
            examples: Batch dict with 'tokens' and 'ner_tags'.
            tokenizer: HuggingFace tokenizer.

        Returns:
            Tokenized batch with aligned 'labels'.
        """
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_offsets_mapping=False,
        )

        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned_labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != previous_word_id:
                    if word_id < len(labels):
                        aligned_labels.append(labels[word_id])
                    else:
                        aligned_labels.append(-100)
                else:
                    if word_id < len(labels):
                        label = labels[word_id]
                        if self.id2label.get(label, "O").startswith("B-"):
                            entity_type = self.id2label[label][2:]
                            i_tag = f"I-{entity_type}"
                            aligned_labels.append(self.label2id.get(i_tag, -100))
                        else:
                            aligned_labels.append(label)
                    else:
                        aligned_labels.append(-100)

                previous_word_id = word_id

            all_labels.append(aligned_labels)

        tokenized["labels"] = all_labels
        return tokenized

    def prepare_dataset(
        self,
        dataset: Dataset,
        model_name: str = "bert-base-uncased",
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        """Tokenize and split dataset for training.

        Args:
            dataset: Raw HuggingFace Dataset with tokens and ner_tags.
            model_name: Pretrained model name for tokenizer.
            test_size: Fraction for test split.
            val_size: Fraction for validation split.
            seed: Random seed.

        Returns:
            DatasetDict with 'train', 'validation', 'test' splits.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        train_val = split_dataset["train"].train_test_split(
            test_size=val_size / (1 - test_size), seed=seed
        )

        dataset_dict = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": split_dataset["test"],
        })

        tokenized = dataset_dict.map(
            lambda examples: self.tokenize_and_align_labels(examples, self.tokenizer),
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
        )

        logger.info(
            f"Prepared dataset: train={len(tokenized['train'])}, "
            f"val={len(tokenized['validation'])}, test={len(tokenized['test'])}"
        )
        return tokenized

    def get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """Compute entity counts and label distribution statistics.

        Args:
            dataset: Dataset with 'ner_tags' column.

        Returns:
            Dictionary with entity counts, label distribution, and sentence length stats.
        """
        all_tags = []
        sentence_lengths = []
        entity_counts = Counter()

        for example in dataset:
            tags = example["ner_tags"]
            tokens = example["tokens"]
            sentence_lengths.append(len(tokens))

            for tag_id in tags:
                label = self.id2label.get(tag_id, "O")
                all_tags.append(label)
                if label.startswith("B-"):
                    entity_type = label[2:]
                    entity_counts[entity_type] += 1

        tag_distribution = Counter(all_tags)

        return {
            "num_sentences": len(dataset),
            "avg_sentence_length": float(np.mean(sentence_lengths)),
            "max_sentence_length": int(np.max(sentence_lengths)),
            "min_sentence_length": int(np.min(sentence_lengths)),
            "total_tokens": sum(sentence_lengths),
            "entity_counts": dict(entity_counts),
            "label_distribution": dict(tag_distribution),
            "num_entity_types": len(entity_counts),
        }

    def create_sample_data(self, num_sentences: int = 500, seed: int = 42) -> Dataset:
        """Generate sample NER data for demonstration and testing.

        Creates synthetic sentences with annotated person names, organizations,
        locations, and miscellaneous entities.

        Args:
            num_sentences: Number of sentences to generate.
            seed: Random seed for reproducibility.

        Returns:
            HuggingFace Dataset with 'tokens' and 'ner_tags' columns.
        """
        rng = random.Random(seed)

        persons = [
            ["John", "Smith"], ["Marie", "Curie"], ["Albert", "Einstein"],
            ["Ada", "Lovelace"], ["Alan", "Turing"], ["Grace", "Hopper"],
            ["Elon", "Musk"], ["Tim", "Cook"], ["Satya", "Nadella"],
            ["Sundar", "Pichai"], ["Warren", "Buffett"], ["Jeff", "Bezos"],
            ["Lisa", "Su"], ["Jensen", "Huang"], ["Demis", "Hassabis"],
        ]
        organizations = [
            "Google", "Microsoft", "Apple", "Amazon", "Meta",
            "NVIDIA", "OpenAI", "DeepMind", "Tesla", "SpaceX",
            "Stanford", "MIT", "Harvard", "NASA", "CERN",
            "WHO", "UNESCO", "Reuters", "Bloomberg", "DARPA",
        ]
        locations = [
            "London", "Paris", "Tokyo", "Berlin", "Sydney",
            "New York", "San Francisco", "Beijing", "Mumbai", "Toronto",
            "California", "Texas", "Germany", "Japan", "Australia",
        ]
        misc_entities = [
            "Nobel Prize", "World Cup", "Olympics", "Grammy",
            "Python", "Linux", "COVID-19", "GDP", "UEFA",
            "Emmy", "Pulitzer", "NASDAQ", "Bitcoin", "Ethereum",
        ]

        templates = [
            (["said", "that", "the", "company", "is", "expanding", "."],
             [None, None, None, None, None, None, None]),
            (["announced", "a", "new", "partnership", "with"],
             [None, None, None, None, None]),
            (["reported", "record", "earnings", "in", "the", "latest", "quarter", "."],
             [None, None, None, None, None, None, None, None]),
            (["was", "born", "in"], [None, None, None]),
            (["is", "headquartered", "in"], [None, None, None]),
            (["received", "the"], [None, None]),
            (["is", "the", "CEO", "of"], [None, None, None, None]),
            (["founded"], [None]),
            (["works", "at"], [None, None]),
            (["traveled", "to"], [None, None]),
        ]

        all_tokens = []
        all_tags = []

        for _ in range(num_sentences):
            tokens = []
            tags = []

            if rng.random() < 0.7:
                person = rng.choice(persons)
                tokens.extend(person)
                tags.append(self.label2id["B-PER"])
                for _ in range(len(person) - 1):
                    tags.append(self.label2id["I-PER"])

            template_tokens, _ = rng.choice(templates)
            tokens.extend(template_tokens)
            tags.extend([self.label2id["O"]] * len(template_tokens))

            entity_type = rng.choice(["org", "loc", "misc", "person"])
            if entity_type == "org":
                org = rng.choice(organizations)
                org_tokens = org.split()
                tokens.extend(org_tokens)
                tags.append(self.label2id["B-ORG"])
                for _ in range(len(org_tokens) - 1):
                    tags.append(self.label2id["I-ORG"])
            elif entity_type == "loc":
                loc = rng.choice(locations)
                loc_tokens = loc.split()
                tokens.extend(loc_tokens)
                tags.append(self.label2id["B-LOC"])
                for _ in range(len(loc_tokens) - 1):
                    tags.append(self.label2id["I-LOC"])
            elif entity_type == "misc":
                misc = rng.choice(misc_entities)
                misc_tokens = misc.split()
                tokens.extend(misc_tokens)
                tags.append(self.label2id["B-MISC"])
                for _ in range(len(misc_tokens) - 1):
                    tags.append(self.label2id["I-MISC"])
            else:
                person = rng.choice(persons)
                tokens.extend(person)
                tags.append(self.label2id["B-PER"])
                for _ in range(len(person) - 1):
                    tags.append(self.label2id["I-PER"])

            if rng.random() < 0.4:
                filler = rng.choice(["in", "at", "near", "from"])
                tokens.append(filler)
                tags.append(self.label2id["O"])
                loc = rng.choice(locations)
                loc_tokens = loc.split()
                tokens.extend(loc_tokens)
                tags.append(self.label2id["B-LOC"])
                for _ in range(len(loc_tokens) - 1):
                    tags.append(self.label2id["I-LOC"])

            if not tokens[-1].endswith("."):
                tokens.append(".")
                tags.append(self.label2id["O"])

            all_tokens.append(tokens)
            all_tags.append(tags)

        dataset = Dataset.from_dict({
            "tokens": all_tokens,
            "ner_tags": all_tags,
        })

        logger.info(f"Created sample dataset with {num_sentences} sentences")
        return dataset
