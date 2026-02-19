# Named Entity Recognition with Transformers

A production-ready NER system built on fine-tuned BERT and RoBERTa models. Supports training on custom datasets with BIO tagging, evaluation with seqeval metrics, and real-time inference through a FastAPI service. Designed for domain-specific entity extraction tasks.

## Overview

This project provides an end-to-end pipeline for Named Entity Recognition:
- Fine-tune pretrained transformer models on custom NER datasets
- Evaluate with standard sequence labeling metrics (per-entity precision, recall, F1)
- Deploy as a REST API with batch inference support
- Handle sub-word tokenization alignment for accurate entity boundary detection

## Architecture

```
Raw Text
    |
    v
Tokenizer (WordPiece/BPE) ──> Sub-word Tokens
    |
    v
BERT/RoBERTa Encoder ──> Contextual Embeddings
    |
    v
Token Classification Head ──> BIO Tag Predictions
    |
    v
Post-Processing ──> Entity Spans with Labels & Confidence
```

## Project Structure

```
nlp-ner-transformers/
├── configs/
│   └── config.yaml
├── src/
│   ├── data_processor.py     # Data loading and tokenization
│   ├── model.py              # Model training and evaluation
│   ├── inference.py          # Inference pipeline
│   └── api.py                # FastAPI serving endpoint
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Train on Custom Data

```python
from src.data_processor import NERDataProcessor
from src.model import NERModel

# Load and prepare data
processor = NERDataProcessor(label_list=[
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC"
])

# Load CoNLL format data or use sample data
dataset = processor.create_sample_data()
tokenized = processor.prepare_dataset(dataset, model_name="bert-base-uncased")

# Train model
ner_model = NERModel(
    model_name="bert-base-uncased",
    label_list=processor.label_list,
    label2id=processor.label2id,
    id2label=processor.id2label,
)
ner_model.train(tokenized, num_epochs=5, batch_size=16)
```

### Inference

```python
from src.inference import NERInference

inference = NERInference(model_path="./models/ner_model")
entities = inference.predict_text("Apple CEO Tim Cook announced new products in Cupertino.")
for entity in entities:
    print(f"{entity['text']} -> {entity['label']} ({entity['confidence']:.3f})")
```

### API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Google was founded by Larry Page and Sergey Brin in Mountain View."}'
```

## Supported Entity Types

| Label | Description |
|-------|-------------|
| PER   | Person names |
| ORG   | Organizations |
| LOC   | Locations |
| MISC  | Miscellaneous entities |

Custom entity types can be added through the configuration file.

## Evaluation

The system reports standard sequence labeling metrics using the seqeval library:
- **Precision**: Fraction of predicted entities that are correct
- **Recall**: Fraction of true entities that are detected
- **F1 Score**: Harmonic mean of precision and recall
- Metrics are reported per entity type and as micro/macro averages

## Configuration

See `configs/config.yaml` for model selection, training hyperparameters, and label definitions.

## License

MIT License
