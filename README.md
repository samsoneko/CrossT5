# CrossT5

Project that aims to combine the PTAE model with a large language model, like T5.
CrossT5 is based on:

Paired Transformed Autoencoders (PTAE)

Last updated: 26 May 2023.
This code has been adapted from Copyright (c) 2018, Tatsuro Yamada <<yamadat@idr.ias.sci.waseda.ac.jp>>
Original repository: https://github.com/ogata-lab/PRAE/
Copyright (c) 2023, Ozan Özdemir <<ozan.oezdemir@uni-hamburg.de>>

Text-to-Text Transfer Transformer (T5), as part of the huggingface transformers package

Variant: t5-small
Last updated: 10 June 2023.
Original repository: https://github.com/google-research/text-to-text-transfer-transformer
Copyright (c) Google

## Requirements
- Python 3
- Pytorch
- NumPy
- Tensorboard

## Implementation
CrossT5 - Pytorch Implementation

Based on:
- Paired Transformed Autoencoders (PTAE) - Pytorch Implementation (Modified)
- Text-to-Text Transfer Transformer (T5) - Pytorch Implementation

For the CrossT5 model to work, the transformers package (which provides the T5 checkpoints) needs to be installed to a location from where it can be imported.
This can be done manually with the following commands:
```
$ git clone https://github.com/huggingface/transformers.git
$ cd transformers
$ pip install -e .
```

## Example
Training the model:
```
$ cd src
$ python main_ptae.py
```

Evaluating the model:
```
$ cd src
$ python inference.py
```

Evaluating the language robustness of the model:
```
$ cd src
$ python inference_robustness.py
```

In use for this project:
- For training the model:
    - main_ptae.py: trains the CrossT5 model
    - ptae.py: defines the CrossT5 architecture
    - t5.py: handles functions related to the T5 instance
    - crossmodal_transformer: defines the Crossmodal Transformer
    - config.py: training and model configurations
    - data_util.py: provides functions for reading the data
    - dataset.py: defines the dataset structure
- For evaluating the model:
    - inference.py: inference time implementation for CrossT5, handles language
    - proprioception_eval.py: inference time implementation for CrossT5, handles action (gets called by inference.py)
    - inference_robustness.py: robustness inference time implementation for CrossT5, handles language
    - proprioception_eval_robustness.py: robustness inference time implementation for CrossT5, handles action (gets called by inference_robustness.py)