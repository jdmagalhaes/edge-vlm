## Project:
Edge Vision Language Model (DotVLM) deployed to the Coral Dev Board Micro.

## Structure:
'model.py' - Original model before any changes.

'model_for_coral.py' - TFLite and QAT friendly version of the same model which was deployed to the Coral Dev Board Micro. 

'dataset.txt' - URLs to the datasets used to train the models.

'dotvlm_coral' directory:
	'apps/' - Code running on the board that loads all components of the model successfully.
	'models/' - All model components deployed to the board.

## Datasets:

1. The Cauldron (https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
	- Localized Narratives
	- tallyQA

2. GQA (https://cs.stanford.edu/people/dorarad/gqa/about.html)

3. COCO - captions (https://cocodataset.org/)

4. VisualGenome - Visual Question Answers (https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)	

The code demonstrates that it was possible to load all components of the model but does not include the autoregressive answer generation step.
Testing would require following the Coral Dev Board Micro documentation: https://www.coral.ai/docs/dev-board-micro/get-started