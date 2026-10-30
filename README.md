(In progress)

## Edge Vision Language Model (DotVLM)
DotVLM is an autoregressive, ~50M-parameter edge vision-language model.

It was originally intended to run on the Coral Dev Micro. However, the board has several limitations (e.g., no support for operations such as LayerNorm or residual connections), which made direct deployment impossible. Adapting the model to fit these limitations severely degraded its performance, and end-to-end deployment on the board was never completed.

This repo therefore archives the best working version of the pre-adapted DotVLM model.

Below are some good and some less good output examples:

## Model Architecture
The model's architecture consists of four main modules:

- **Vision Encoder:** Pretrained MobileNetV4 Hybrid backbone extracts early, fine, and coarse visual features, each projected to 512 dimensions and given 2D positional coordinates.
- **Text Encoder:** Embedded questions pass through depthwise separable convolution layers (kernel sizes 1, 3, 5, 7), producing per-token embeddings and a pooled question embedding.
- **Question-Conditioned Pooling:** The pooled question embedding generates learned queries that cross-attend over each visual scale, compressing the image into 96 question-relevant visual tokens.
- **Convolutional Decoder:** A causal temporal convolution network (TCN) of 7 dilated residual blocks run over the visual, question, and answer tokens, and a linear head predicts the answer tokens (teacher forcing in training, autoregressive at inference).

## Datasets and Vocabulary
The model was trained from scratch using the following datasets and training/validation splits:

| Dataset | Train images | Val images |
|---|---:|---:|
| [Localized Narratives](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/localized_narratives) ([The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)) | 180,000 | 18,000 |
| [TallyQA](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/tallyqa) ([The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)) | 12,000 | 1,200 |
| [COCO captions 2017](https://cocodataset.org/) | 118,287 | 5,000 |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html) | 72,140 | 8,000 |
| [Visual Genome QA](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | 79,477 | 8,000 |
| **Total images** | **461,904** | **40,200** |
| **QA samples** | **1,957,200** | **169,702** |
| **Batches** (batches of 64 QA samples) | **30,581** | **2,652** |

Counts are for the main training run (up to 10 QA pairs per image). Fine-tuning uses up to 3.

### Unified Record Format
Every source is turned into the same **record**, so all datasets go through one pipeline. A record holds one image and its QA pairs:

```json
{
    "image_path": "/path/to/image.jpg",
    "texts": [
        {"user": "What is in the image?", "assistant": "A person riding a horse."},
        {"user": "What is the person wearing?", "assistant": "A helmet and a jacket."}
    ]
}
```

Each record is expanded into one (image, question, answer) sample per QA pair, with at most 10 pairs per image.

**The Cauldron** (Localized Narratives and TallyQA) already comes with questions and answers. `download.py` streams the subsets from HuggingFace, saves the images as JPEGs, and writes them directly as records.

**COCO captions, GQA and Visual Genome QA** were first converted into a shared entry format. Each file is a list of groups, one per image, and each group is a list of QA entries, as shown below. `dataset.py` then matches each group to its image file and turns it into a record.

```json
[
  [
    {
      "user": "Describe what you observe.",
      "assistant": "An office cubicle with four different types of computers.",
      "source": "coco_captions",
      "image_id": 190236
    },
    {
      "user": "Provide a description.",
      "assistant": "The home office space seems to be very cluttered.",
      "source": "coco_captions",
      "image_id": 190236
    }
  ]
]
```

COCO captions come without questions, so each caption was paired with one of a small set of generic prompts (e.g. "Describe this image.", "What do you see?", "Provide a description.", "What can you tell me about this image?"). GQA and Visual Genome QA keep their original questions. The preprocessed files are available at the HuggingFace repo [edge-vlm](https://huggingface.co/datasets/jdmagalhaes/edge-vlm). The images themselves are not included, so they must be downloaded from their original sources.

### Vocabulary
The model uses a reduced vocabulary of 8,000 tokens, including the required special tokens. 

It is a subset of the [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) tokenizer (~49k tokens): the 8,000 most frequent tokens counted over a selection of The Cauldron's configs. 

Text is tokenized with the SmolVLM tokenizer, each token ID is remapped to the reduced vocabulary, and any token outside it becomes `<unk>`.

The vocabulary and mappings between the original and reduced IDs are available at the HuggingFace repo [edge-vlm](https://huggingface.co/datasets/jdmagalhaes/edge-vlm).

The vocabulary file is a json containing:

- `vocab`: the 8,000 tokens
- `old_to_new`: SmolVLM token ID → reduced ID
- `new_to_old`: reduced ID → SmolVLM token ID
- `vocab_size`: 8000
- `reserved_tokens`: the 4 special tokens (`<pad>`=0, `<eos>`=1, `<bos>`=2, `<unk>`=3)

## Usage

### Requirements
- [uv](https://docs.astral.sh/uv/) and Python 3.13 (developed and tested with 3.13; newer versions should work)

Install the dependencies:

```bash
uv sync
```

### Setup
Run all commands below from the `src/` folder (`cd src`).

1. Set `DATA_ROOT` in `config.py` to the folder where your datasets will live.
2. Download the COCO, GQA and Visual Genome images from their original sources and arrange them like this:

   ```
   DATA_ROOT/other_datasets/
   ├── COCO/images/train2017, val2017
   ├── GQA/images
   └── VGREGQA/images/VG_100K, VG_100K_2
   ```

3. Download the json and vocabulary files from the [edge-vlm](https://huggingface.co/datasets/jdmagalhaes/edge-vlm) HF repo and put the `json_data/` and `vocab/` folders in `src/`.
4. Download The Cauldron subsets (only has to be run once):

   ```bash
   uv run python download.py
   ```

   They are saved to `DATA_ROOT/cauldron_subsets`.

### Training and Fine-tuning
To train, run:

```bash
uv run python train.py
```

The best model is saved to `./checkpoints/best_model.pt`.

Next, fine-tune on all datasets with fewer QA pairs per image (3 instead of 10). This gives Localized Narratives more weight (since it only has 1 QA pair per image) and produces more verbose descriptions. It starts from `best_model.pt` and is needed for the default inference setup:

```bash
uv run python finetune.py
```

The best fine-tuned model is saved to `./checkpoints/best_model_ft.pt`. Nothing is saved if fine-tuning doesn't improve on the starting validation loss.

### Inference
Set `image_path` in `inference.py` to the image you want to test (`test_data/dog1.jpg` by default), then run:

```bash
uv run python inference.py
```

By default, inference requires the fine-tuned checkpoint (`./checkpoints/best_model_ft.pt`), so run `finetune.py` first. To use the base model instead, set `checkpoint_path` in `load_model` to `cfg.CHECKPOINT_PATH`.

Sample image: [Chihuahua dog 1.jpg](https://commons.wikimedia.org/wiki/File:Chihuahua_dog_1.jpg) by Davidstern, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed.en).