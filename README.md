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
The model was trained from scratch using the following datasets:

- [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
	- Localized Narratives
	- tallyQA
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)
- [COCO - captions](https://cocodataset.org/)
- [VisualGenome - Visual Question Answers](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)	