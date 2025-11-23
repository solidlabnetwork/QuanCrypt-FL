# ... [IMPORTS REMAIN UNCHANGED] ...
import os
import csv
import time
import copy
import tenseal as ts
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import inversefed
from utils.gi_attack import gia
from utils.misc_utils import *
from utils.QuanCryptFL_update import client_train
from utils.Nets import *
from utils.Prune import prune_model, set_weight_by_mask
from utils.quant_utils import quantize_tensor, dequantize_tensor
from utils.HE_utils import encrypt_weights, decrypt_weights, aggregate_encrypted_updates
from utils.Options import parse_args

args = parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
set_seed(42)

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2 ** 40

model_name, n_clients, n_local_epochs = args.model_name, args.n_clients, args.n_local_epochs
num_rounds, learning_rate, weight_decay = args.num_rounds, args.learning_rate, args.weight_decay
num_bits, clip_factor, reconfig_freq, batch_size = args.num_bits, args.clip_factor, args.reconfig_freq, args.batch_size
lambda_param = args.lambda_param
perform_attack = args.perform_attack
num_bits=args.num_bits
target_pruning_rate=args.target_pruning_rate

checkpoint_dir, output_dir = args.checkpoint_dir, args.output_dir
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

csv_filename = args.csv_filename if args.csv_filename else (
    f"{output_dir}/QuanCryptFL-{model_name}-{args.dataset}-{args.batch_size}-{args.partition}-"
    f"clip_factor({clip_factor})-lambda_param({lambda_param})-target_pruning_rate({args.target_pruning_rate})-num_bits({args.num_bits})-clients({n_clients}).csv"
)

if args.dataset == 'HAM10000':
    train_dataset, val_dataset, test_dataset = prepare_data(args)
else:
    full_dataset, test_dataset = prepare_data(args)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices = list(range(len(full_dataset)))
    train_split, val_split = random_split(train_indices, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_dataset = torch.utils.data.Subset(full_dataset, train_split)
    val_dataset = torch.utils.data.Subset(full_dataset, val_split)

if args.partition == 'IID':
    if args.dataset == 'CIFAR10':
        dict_users = cifar10_iid(train_dataset, n_clients)
    elif args.dataset == 'CIFAR100':
        dict_users = cifar10_iid(train_dataset, n_clients)
    elif args.dataset == 'MNIST':
        dict_users = mnist_iid(train_dataset, n_clients)
    elif args.dataset == 'HAM10000':
        dict_users = ham_iid(train_dataset, n_clients)
else:
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        subset_targets = [full_dataset[i][1] for i in train_split]
        target_based_dataset = [(i, subset_targets[i]) for i in range(len(subset_targets))]
        dict_users = cifar_noniid(target_based_dataset, n_clients)
        dict_users = {i: [train_split[idx] for idx in user_idxs] for i, user_idxs in dict_users.items()}
    elif args.dataset == 'MNIST':
        dict_users = mnist_noniid(train_dataset, n_clients)
    elif args.dataset == 'HAM10000':
        dict_users = ham_noniid(train_dataset, n_clients)

dataset_for_partition = train_dataset if args.dataset == 'HAM10000' else full_dataset
client_dataloaders = []
valid_clients = []

for i in range(n_clients):
    if len(dict_users[i]) > 0:
        client_dataloaders.append(DataLoader(DatasetSubset(dataset_for_partition, dict_users[i]), batch_size=batch_size, shuffle=True))
        valid_clients.append(i)
    else:
        print(f"[WARNING] Skipping Client {i}: No data assigned.")

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for i, dataloader in zip(valid_clients, client_dataloaders):
    total_samples = sum(len(batch[0]) for batch in dataloader)
    print(f"Client {i} has {total_samples} training samples.")

models_dict = {'CNNMNIST': CNNMNIST, 'AlexNet': AlexNet, 'ResNet18': ResNet18}
model_type = models_dict.get(model_name)
global_model = model_type().to(device) if model_type else None

checkpoint_path1 = os.path.join(checkpoint_dir, f'QuanCryptFL-{model_name}-{args.dataset}-{args.batch_size}-{args.partition}-clip_factor({clip_factor})-lambda_param({lambda_param})-target_pruning_rate({args.target_pruning_rate})-num_bits({args.num_bits})-clients({n_clients})_1.pth')
checkpoint_path2 = os.path.join(checkpoint_dir, f'QuanCryptFL-{model_name}-{args.dataset}-{args.batch_size}-{args.partition}-clip_factor({clip_factor})-lambda_param({lambda_param})-target_pruning_rate({args.target_pruning_rate})-num_bits({args.num_bits})-clients({n_clients})_2.pth')

if os.path.exists(checkpoint_path1):
    checkpoint = torch.load(checkpoint_path1)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    start_round = checkpoint['round']
    test_accuracy_list = checkpoint['test_accuracy_list']
    train_accuracy_list = checkpoint['train_accuracy_list']
    results = checkpoint['results']
else:
    start_round, test_accuracy_list, train_accuracy_list, results = 0, [], [], []

early_stopping = EarlyStopping(patience=args.patience)
best_accuracy = 0.0
print(f"Starting from round {start_round + 1}")

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Train Accuracy", "Train Loss", "Test Accuracy", "Test Loss", "Val Accuracy", "Val Loss",
                         "Round Training Time", "Checkpoint Times", "Quantization Times", "Encryption Times",
                         "Decryption Times", "Aggregation Times", "Inference Times", "Total Training Time"])

round_times, checkpoint_times, encryption_times, decryption_times, aggregation_times, inference_times, quantization_times = [], [], [], [], [], [], []
reconfig_mask = None

for round in range(start_round, num_rounds):
    round_start_time = time.time()
    local_models, encrypted_local_updates = [], []
    global_state_dict = global_model.state_dict()
    total_encryption_time, total_decryption_time = 0, 0

    pruning_rate = max(0, (round - args.effective_round) / (args.target_round - args.effective_round)) * (args.target_pruning_rate - args.initial_pruning_rate) + args.initial_pruning_rate

    for idx, client_id in enumerate(valid_clients):
        client_model = copy.deepcopy(global_model)
        client_model.load_state_dict(global_state_dict)
        client_update = client_train(args=args, device=device, lr=learning_rate, weight_decay=weight_decay, dataloader=client_dataloaders[idx])
        client_state_dict, _, pruning_mask = client_update.train(client_model, round, reconfig_mask)
        local_models.append(client_state_dict)

        clipped_client_state_dict = clip_model_update(client_state_dict, clip_factor)
        quant_start_time = time.time()
        quantized_update, scale_zero_point_dict = {}, {}
        for key in clipped_client_state_dict.keys():
            q_x, scale = quantize_tensor(clipped_client_state_dict[key], num_bits)
            quantized_update[key] = q_x
            scale_zero_point_dict[key] = scale
        quantization_times.append(time.time() - quant_start_time)

        enc_start_time = time.time()
        encrypted_client_update = encrypt_weights(quantized_update, context, encryption_times)
        encrypted_local_updates.append((encrypted_client_update, scale_zero_point_dict))
        total_encryption_time += time.time() - enc_start_time

    encryption_times.append(total_encryption_time)

    agg_start_time = time.time()
    aggregated_encrypted_update = aggregate_encrypted_updates([x[0] for x in encrypted_local_updates], context)
    aggregation_times.append(time.time() - agg_start_time)

    dec_start_time = time.time()
    decrypted_update = decrypt_weights(aggregated_encrypted_update, context, global_model, decryption_times)
    total_decryption_time += time.time() - dec_start_time
    decryption_times.append(total_decryption_time)

    for key in decrypted_update.keys():
        scale = encrypted_local_updates[0][1][key]
        dequantized_avg = dequantize_tensor(scale, decrypted_update[key])
        if global_state_dict[key].dtype == torch.long:
            global_state_dict[key] = (1 - lambda_param) * global_state_dict[key].long() + lambda_param * dequantized_avg.long()
        else:
            global_state_dict[key] += (1 / len(valid_clients)) * (dequantized_avg - global_state_dict[key])

            global_state_dict[key] = (1 - lambda_param) * global_state_dict[key] + lambda_param * dequantized_avg
    global_model.load_state_dict(global_state_dict)

    # if reconfig_mask is not None:
    #     set_weight_by_mask(global_model, mask=reconfig_mask)
    # if (round + 1) % reconfig_freq == 0:
    #     is_resnet = model_name.lower() == 'resnet18'
    #     reconfig_mask = prune_model(model=global_model, sparsity=pruning_rate, is_resnet=is_resnet, include_fc=True, include_bias=False)

    inf_start_time = time.time()
    test_loss, test_accuracy = validate(global_model, test_dataloader, nn.CrossEntropyLoss())
    inference_times.append(time.time() - inf_start_time)
    val_loss, val_accuracy = validate(global_model, val_loader, nn.CrossEntropyLoss())

    test_accuracy_list.append(test_accuracy)
    train_loss, train_accuracy = validate(global_model, client_dataloaders[0], nn.CrossEntropyLoss()) if len(client_dataloaders) > 0 else (0.0, 0.0)
    train_accuracy_list.append(train_accuracy)
    results.append([round + 1, train_accuracy, train_loss, test_accuracy, test_loss, val_accuracy, val_loss])

    print("*******************************************************************************************")
    print(f'Round {round + 1}, Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.4f}')
    print(f'Round {round + 1}, Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}')
    print(f'Round {round + 1}, Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}')
    print("*******************************************************************************************")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(global_model.state_dict(), checkpoint_path2)
        print(f"Best model saved with validation accuracy: {val_accuracy:.2f}%")

    early_stopping(val_loss, global_model)
    if early_stopping.early_stop:
        print("Early stopping.")
        early_stopping.early_stop = False
        early_stopping.counter = 0

    save_start_time = time.time()
    torch.save({
        'round': round + 1,
        'model_state_dict': global_model.state_dict(),
        'test_accuracy_list': test_accuracy_list,
        'train_accuracy_list': train_accuracy_list,
        'results': results
    }, checkpoint_path1)
    checkpoint_times.append(time.time() - save_start_time)
    round_times.append(time.time() - round_start_time)

    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round + 1, train_accuracy, train_loss, test_accuracy, test_loss, val_accuracy, val_loss,
                         round_times[-1], checkpoint_times[-1], quantization_times[-1], encryption_times[-1],
                         decryption_times[-1], aggregation_times[-1], inference_times[-1], time.time() - round_start_time])

print(f"Total Training Time: {sum(round_times):.2f} seconds")
print(f"Total Checkpoint Saving Time: {sum(checkpoint_times):.2f} seconds")
print(f"Total Quantization Time: {sum(quantization_times):.2f} seconds")
print(f"Total Encryption Time: {sum(encryption_times):.2f} seconds")
print(f"Total Decryption Time: {sum(decryption_times):.2f} seconds")
print(f"Total Aggregation Time: {sum(aggregation_times):.2f} seconds")
print(f"Total Inference Time: {sum(inference_times):.2f} seconds")

if perform_attack:
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, device=device)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, device=device)[:, None, None]
    if os.path.exists(checkpoint_path1):
        checkpoint = torch.load(checkpoint_path1)
        global_model.load_state_dict(checkpoint['model_state_dict'])
    gia(global_model, train_dataset, dict_users, dm, ds, device)
