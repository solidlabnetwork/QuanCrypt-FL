import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="QuanCrypt-FL: Quantized Homomorphic Encryption with Pruning for Secure and Efficient Federated Learning"
    )

    parser.add_argument('--device', type=str, default="GPU", choices=["GPU", "CPU"],
                        help='Device to be used for Federated Learning (GPU or CPU)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100','HAM10000'],
                        help='Dataset to use (MNIST, CIFAR10, CIFAR100, HAM10000)')
    parser.add_argument('--partition', type=str, default='nonIID', choices=['IID', 'nonIID'],
                        help='Data partitioning strategy (IID or nonIID)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for dataset storage')
    parser.add_argument('--model_name', type=str, default='CNNMNIST',
                        choices=['CNNMNIST', 'AlexNet', 'ResNet18','HAMAlexNet'],
                        help='Model to use (CNNMNIST, AlexNet, ResNet18)')
    parser.add_argument('--n_clients', type=int, default=100,
                        help='Number of clients for federated learning')
    parser.add_argument('--n_local_epochs', type=int, default=1,
                        help='Number of local epochs per client')
    parser.add_argument('--num_rounds', type=int, default=100,
                        help='Number of communication rounds')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for client-side training')
    parser.add_argument('--num_bits', type=int, default=8,
                        help='Number of bits for quantization')
    parser.add_argument('--clip_factor', type=float, default=3.0,
                        help='Clip factor for model updates')
    parser.add_argument('--initial_pruning_rate', type=float, default=0.2,
                        help='Initial pruning rate')
    parser.add_argument('--target_pruning_rate', type=float, default=0.5,
                        help='Target pruning rate')
    parser.add_argument('--effective_round', type=int, default=40,
                        help='Round to start pruning')
    parser.add_argument('--target_round', type=int, default=100,
                        help='Round to reach target pruning rate')
    parser.add_argument('--reconfig_freq', type=int, default=50,
                        help='Frequency of reconfiguration in rounds')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='result',
                        help='Directory to save output results')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--csv_filename', type=str, default=None,
                        help='CSV filename for logging training metrics')
    parser.add_argument('--lambda_param', type=float, default=1.0,
                        help='Lambda parameter for model aggregation smoothing')
    parser.add_argument('--perform_attack', type=bool, default=False,
                        help='Enable reconstruction attack if True')

    return parser.parse_args()