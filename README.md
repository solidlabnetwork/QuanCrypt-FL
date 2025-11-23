# QuanCrypt-FL: Quantized Homomorphic Encryption with Pruning for Secure and Efficient Federated Learning

## Overview
QuanCrypt-FL is an innovative approach designed to enhance security in federated learning by combining quantization and pruning strategies. This integration strengthens the defense against adversarial attacks while also lowering computational overhead during training. Additionally, the approach includes a mean-based clipping mechanism to address potential quantization overflows and errors. The combination of these techniques results in a communication-efficient FL framework that prioritizes privacy without significantly affecting model accuracy, thereby boosting computational performance and resilience to attacks.

## Features
- Federated Learning (FL) setup with customizable client count and local training parameters.
- CKKS scheme-based homomorphic encryption for secure aggregation.
- Quantization for efficient communication and storage.
- Pruning to optimize the model and reduce communication overhead.
- Support for CIFAR-10, CIFAR-100, and MNIST datasets with IID and non-IID partitioning.
- Integration of early stopping to enhance training efficiency.
- Comprehensive logging and result storage in CSV files.
- Inverse federated attacks to test model robustness.
- Customizable training, quantization, and encryption parameters.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/juealcs/QuanCryptFL.git
    cd QuanCryptFL
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have PyTorch and TenSEAL installed. For installation details, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [TenSEAL documentation](https://github.com/OpenMined/TenSEAL).

## Requirements
- Python 3.7+
- PyTorch
- TenSEAL
- Torchvision
- CUDA (for GPU support)
- [InverseFed](https://github.com/JonasGeiping/invertinggradients) 


## Arguments
QuanCryptFL supports a variety of arguments for customizing the FL setup, training, and evaluation. Below is a list of available arguments and their descriptions:

| Argument Name               | Description                                                             | Default Value     |
|-----------------------------|-------------------------------------------------------------------------|--------------------|
| `--device`                  | Device to be used for Federated Learning (GPU or CPU)                   | `GPU`              |
| `--dataset`                 | Dataset to use (`MNIST`, `CIFAR10`, `CIFAR100`)                         | `CIFAR100`         |
| `--partition`               | Data partitioning strategy (`IID` or `nonIID`)                          | `IID`              |
| `--data_dir`                | Directory for dataset storage                                           | `./data`           |
| `--model_name`              | Model to use (`CNNMNIST`, `CNNCIFAR`, `AlexNet`, `ResNet18`)            | `ResNet18`         |
| `--n_clients`               | Number of clients for federated learning                                | `50`               |
| `--n_local_epochs`          | Number of local epochs per client                                       | `1`                |
| `--num_rounds`              | Number of communication rounds                                          | `300`              |
| `--learning_rate`           | Learning rate for the optimizer                                         | `0.001`            |
| `--weight_decay`            | Weight decay for the optimizer                                          | `0.0001`           |
| `--batch_size`              | Batch size for client-side training                                     | `64`               |
| `--num_bits`                | Number of bits for quantization                                         | `8`                |
| `--clip_factor`             | Clip factor for model updates (1.0-5.0)                                 | `3.0`              |
| `--initial_pruning_rate`    | Initial pruning rate                                                    | `0.2`              |
| `--target_pruning_rate`     | Target pruning rate                                                     | `0.5`              |

## Usage
To run the Vanilla FL script with the desired configurations, use the following command format:
```bash
python Vanilla-FL.py --model_name ResNet18 --dataset CIFAR10 --n_clients 10 --n_local_epochs 1 --num_rounds 300 --batch_size 64 --learning_rate 0.001 --partition IID --lambda_param 0.1 --reconfig_freq 50 --patience 5 --perform_attack True
```
Run the QuanCryptFL mechanism use the following script with the desired configurations:
```bash
python QuanCrypt-FL.py --model_name ResNet18 --dataset CIFAR10 --n_clients 10 --n_local_epochs 1 --num_rounds 300 --batch_size 64 --learning_rate 0.001 --partition IID --clip_factor 0.5 --num_bits 8 --lambda_param 0.1 --reconfig_freq 50 --patience 5 --perform_attack True
```
