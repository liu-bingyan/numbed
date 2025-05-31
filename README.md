# NumBed

This repository contains the implementation for the TRL2025 workshop publication: "Embeddings for Numerical Features Using tanh Activation".

## Code Framework

This codebase is built upon and extends two main frameworks:

1. **TabSurvey Framework** ([source](https://github.com/kathrinse/TabSurvey), MIT License): We utilize and modify their systematic model evaluation framework to conduct comprehensive testing of our approaches. The framework provides standardized implementations for:
   - Data loading and preprocessing
   - Framework for model implementation and training 
   - Model scorer and logging

2. **RTDL Numerical Embeddings** ([source](https://github.com/yandex-research/rtdl-num-embeddings), MIT License): We incorporate their FT-Transformer implementation along with their reported optimal hyperparameters. This includes:
   - FT-Transformer implementation
   - Dataset configuration and downloading
   - Finetuned hyperparameters for base models

We combine these frameworks to test our new numerical embedding methods. Copies of the original licenses can be found in the `licenses/` folder.

## Configuration and Usage

The code is designed to run in a containerized environment. You can use either Docker or Apptainer, with their respective configuration files (`Dockerfile` or `apptainer.def`).

To run experiments, use the `train.py` script with appropriate arguments. Example scripts can be found in the `examples/` directory. The script accepts various command-line arguments that are specified in `utils/parser.py`. These arguments allow you to configure the model selection, dataset choice, embedding type, and other parameters.

Dataset and embedding configurations are stored in the `config/` directory. The experimental scenario (1 or 2) is controlled by the `SOURCE` parameter, which can be set in the example configurations. Training logs and results output are controlled by various 'save_*' parameters, allowing you to specify what information should be saved during training.

The numerical embedding implementations can be found in `models/basemodule_embedding.py`. The framework is designed to be extensible - you can add new datasets by modifying `utils/load_data.py` and new models by adding implementations to the `models/` directory.

## Results

Experimental results are organized in two directories: `report-scenario1/` and `report-scenario2/`. Each directory contains the corresponding performance metrics.
