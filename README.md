<div align="center">

# DetectCodeGPT

[![Conference](https://img.shields.io/badge/Conference-ICSE%202025-brightgreen)](https://icse2025.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/)

</div>

How can we develop zero-shot detection of machine generated codes? Welcome to the repository for the research paper: **"Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers."** Our paper has been accepted to the 47th International Conference on Software Engineering (**ICSE 2025**).

## Table of Contents

- [DetectCodeGPT](#detectcodegpt)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Data Preparation](#data-preparation)
  - [Usage](#usage)
    - [Conducting the Empirical Study](#conducting-the-empirical-study)
    - [Using DetectCodeGPT](#using-detectcodegpt)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Getting Started

### Prerequisites

Experiments are conducted using **Python 3.9.7** on an **Ubuntu 22.04.1** server.

To install all required packages, navigate to the root directory of this project and run:

```bash
pip install -r requirements.txt
```

### Data Preparation

To prepare the datasets used in our study, you will first download human-written code and then generate AI counterparts for comparison.

1. **Download Human Dataset**
   Navigate to the `download-dataset` directory and download the required dataset (e.g., from The Vault). We recommend standardizing inside the `dataset_human` folder.

   ```bash
   python download-dataset/download_dataset.py dataset_human --set function
   ```

2. **Generate AI Code Dataset**
   Navigate to the `code-generation` directory. This script uses the downloaded human dataset prompts to generate corresponding AI code.

   ```bash
   python code-generation/generate.py --path dataset_human --language all --output_dir dataset_ai
   ```

## Usage

### Generating Features

Extract advanced statistical and curvature-based features from the code to distinguish human vs AI written files:

1. Navigate to the `generate-features` directory.
2. Run the feature generation script on the prepared datasets.

   ```bash
   python generate-features/generate_features.py --samples 500 --models all --device cuda
   ```

   This will extract a CSV dataset of numerical features to the `features/` directory.

### Training the MLP Detection Model

To evaluate the AI code detection capability, train a Multi-Layer Perceptron (MLP) on the generated features:

1. Navigate to the `training` directory.
2. Run the Optuna hyperparameter optimization script on the saved features.

   ```bash
   python training/train_mlp.py --features "features/*.csv" --trials 50
   ```

   This will search for the best model architecture and parameters to maximize F1 detection score and output the final model weights to `dl_models/`.

## Acknowledgements

The code is modified based on the original repositories of [DetectGPT](https://github.com/eric-mitchell/detect-gpt/tree/main/) and [DetectLLM](https://github.com/mbzuai-nlp/DetectLLM). We thank the authors for their contributions.

## Citation

If you use DetectCodeGPT in your research, please cite our paper:

```bibtex
@inproceedings{shi2025detectcodegpt,
  title={Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers},
  author={Shi, Yuling and Zhang, Hongyu and Wan, Chengcheng and Gu, Xiaodong},
  booktitle={Proceedings of the 47th International Conference on Software Engineering (ICSE 2025)},
  year={2025},
  organization={IEEE}
}
```
