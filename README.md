(To be completed)

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
| [COCO captions](https://cocodataset.org/) | 118,287 | 5,000 |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html) | 72,140 | 8,000 |
| [Visual Genome QA](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | 79,477 | 8,000 |
| **Total images** | **461,904** | **40,200** |
| **QA samples** | **1,957,200** | **169,702** |
| **Batches** (batches of 64 QA samples) | **30,581** | **2,652** |