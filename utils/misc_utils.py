import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import pickle
import GPUtil
from utils.Options import parse_args
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Parse arguments
args = parse_args()

# Utility to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Clip model updates to avoid instability
def clip_model_update(model_update, clip_factor=2.0):
    clipped_update = {}
    for key, value in model_update.items():
        if not value.is_floating_point():
            value = value.float()
        mean_value = value.abs().mean()
        clipped_value = torch.clamp(value, -clip_factor * mean_value, clip_factor * mean_value)
        clipped_update[key] = clipped_value
    return clipped_update

def clip_model_params(model_state_dict, min_val, max_val):
    clipped_state_dict = {}
    for key, param in model_state_dict.items():
        clipped_state_dict[key] = torch.clamp(param, min=min_val, max=max_val)
    return clipped_state_dict

# Custom CIFAR-10 dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        if self.train:
            self.data, self.labels = [], []
            for i in range(1, 6):
                file = os.path.join(data_dir, 'data_batch_' + str(i))
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    self.data.append(dict[b'data'])
                    self.labels.extend(dict[b'labels'])
            self.data = np.concatenate(self.data)
        else:
            file = os.path.join(data_dir, 'test_batch')
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data = dict[b'data']
                self.labels = dict[b'labels']
        self.data = self.data.reshape((-1, 3, 32, 32)).astype(np.float32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, label

class HAMDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_full_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['label']

def get_ham_data():
    root_dir = "/disk/solidlab-server/lclhome/mmia001/phe/HAM10000"
    csv_path = os.path.join(root_dir, "HAM10000_metadata.csv")
    img_dirs = [
        os.path.join(root_dir, "HAM10000_images_part_1"),
        os.path.join(root_dir, "HAM10000_images_part_2")
    ]

    df = pd.read_csv(csv_path)

    def find_image_path(image_id):
        for directory in img_dirs:
            path = os.path.join(directory, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
        return None

    df['image_full_path'] = df['image_id'].apply(find_image_path)
    df = df[df['image_full_path'].notnull()].reset_index(drop=True)

    label_dict = {label: idx for idx, label in enumerate(df['dx'].unique())}
    df['label'] = df['dx'].map(label_dict)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Step 1: Train/Val/Test Split (80/20 first)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42)

    # Step 2: Train/Val split from the 80% pool
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, stratify=train_val_df['label'], random_state=42)

    # Create Dataset objects
    train_dataset = HAMDataset(train_df, transform)
    val_dataset = HAMDataset(val_df, transform)
    test_dataset = HAMDataset(test_df, transform)

    return train_dataset, val_dataset, test_dataset

# def get_ham_data(data_dir, transform):
#     csv_path = "/disk/solidlab-server/lclhome/mmia001/phe/HAM10000/HAM10000_metadata.csv"
#     img_dirs = [
#         "/disk/solidlab-server/lclhome/mmia001/phe/HAM10000/HAM10000_images_part_1",
#         "/disk/solidlab-server/lclhome/mmia001/phe/HAM10000/HAM10000_images_part_2"
#     ]
#
#     df = pd.read_csv(csv_path)
#
#     def find_image_path(image_id):
#         for directory in img_dirs:
#             path = os.path.join(directory, f"{image_id}.jpg")
#             if os.path.exists(path):
#                 return path
#         return None
#
#     df['image_full_path'] = df['image_id'].apply(find_image_path)
#     df = df[df['image_full_path'].notnull()].reset_index(drop=True)
#
#     label_dict = {label: idx for idx, label in enumerate(df['dx'].unique())}
#     df['label'] = df['dx'].map(label_dict)
#
#     # ✅ Remove hardcoded transform, use provided one instead
#
#     # Step 1: Train/Val/Test Split (80/20 first)
#     train_val_df, test_df = train_test_split(
#         df, test_size=0.2, stratify=df['label'], random_state=42)
#
#     # Step 2: Train/Val split from the 80% pool
#     train_df, val_df = train_test_split(
#         train_val_df, test_size=0.2, stratify=train_val_df['label'], random_state=42)
#
#     # Create Dataset objects
#     train_dataset = HAMDataset(train_df, transform)
#     val_dataset = HAMDataset(val_df, transform)
#     test_dataset = HAMDataset(test_df, transform)
#
#     return train_dataset, val_dataset, test_dataset

# IID partitioning for CIFAR-10 dataset
def cifar10_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar100_iid(dataset, num_users):
    num_items = 5000
    total_samples = len(dataset)

    if total_samples < num_items * 8:
        raise ValueError(f"Not enough data for the first 8 clients. The dataset must have at least {num_items * 8} samples.")

    dict_users = {}
    all_idxs = np.arange(total_samples)
    np.random.shuffle(all_idxs)

    for i in range(8):
        dict_users[i] = set(all_idxs[i * num_items : (i + 1) * num_items])

    for i in range(8, num_users):
        reused_client_idx = i % 8
        dict_users[i] = dict_users[reused_client_idx]

    return dict_users

def cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 250
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Extract labels
    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
        indices = np.array(dataset.indices)
    elif isinstance(dataset, list):
        labels = np.array([label for _, label in dataset])
        indices = np.arange(len(dataset))
    else:
        labels = np.array(dataset.targets)
        indices = np.arange(len(dataset))

    # Sort indices by labels
    idxs_labels = np.vstack((indices, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    sorted_indices = idxs_labels[0, :]

    # Assign shards to users
    shard_per_user = num_shards // num_users
    idx_shard = list(range(num_shards))

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], sorted_indices[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        np.random.shuffle(dict_users[i])

    return dict_users

# IID partitioning for MNIST dataset
def mnist_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# Non-IID partitioning for MNIST dataset (with different labels)
def mnist_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 300
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
        dataset_size = len(dataset.indices)
    else:
        labels = np.array(dataset.train_labels)
        dataset_size = len(dataset)

    idxs = np.arange(dataset_size)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    shard_per_user = int(num_shards / num_users)
    idx_shard = [i for i in range(num_shards)]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        np.random.shuffle(dict_users[i])

    return dict_users
# def ham_dirichlet(dataset, num_users, alpha=0.5, seed=42):
#     rng = np.random.default_rng(seed)
#     n = len(dataset)
#
#     if hasattr(dataset, "targets"):
#         labels = np.array(dataset.targets)
#     elif hasattr(dataset, "labels"):
#         labels = np.array(dataset.labels)
#     else:
#         labels = np.array([dataset[i][1] for i in range(n)])
#
#     classes = np.unique(labels)
#     dict_users = {u: [] for u in range(num_users)}
#
#     for c in classes:
#         idx_c = np.where(labels == c)[0]
#         rng.shuffle(idx_c)
#         # Draw per-user proportions for this class
#         p = rng.dirichlet(alpha * np.ones(num_users))
#         counts = np.floor(p * len(idx_c)).astype(int)
#         # Fix rounding
#         while counts.sum() < len(idx_c):
#             counts[rng.integers(0, num_users)] += 1
#
#         start = 0
#         for u, k in enumerate(counts):
#             if k > 0:
#                 dict_users[u].extend(idx_c[start:start + k])
#                 start += k
#
#     # Finalize
#     dict_users = {u: np.array(rng.permutation(v), dtype='int64') for u, v in dict_users.items()}
#     return dict_users

def ham_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# def ham_noniid(dataset, num_users):
#     num_shards, num_imgs = 100, 100  # 100 shards × 100 imgs ≈ 10,000 total
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     # Get labels and indices
#     if isinstance(dataset, torch.utils.data.Subset):
#         labels = np.array([dataset.dataset[i][1] for i in dataset.indices])
#         indices = np.array(dataset.indices)
#     else:
#         labels = np.array([dataset[i][1] for i in range(len(dataset))])
#         indices = np.arange(len(dataset))
#
#     # Sort indices by label
#     idxs_labels = np.vstack((indices, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     sorted_indices = idxs_labels[0, :]
#
#     # Assign shards
#     shard_per_user = num_shards // num_users
#     idx_shard = list(range(num_shards))
#
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], sorted_indices[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#         np.random.shuffle(dict_users[i])
#
#     return dict_users

import numpy as np
import torch

def ham_noniid(dataset, num_users):
    min_classes = 5          # minimum distinct classes per client
    seed_per_class = 10      # samples per chosen class in seeding
    rng = np.random.default_rng(42)

    n = len(dataset)
    target_per_client = n // num_users  # balanced size

    # --- Get labels & indices ---
    if isinstance(dataset, torch.utils.data.Subset):
        idxs = np.array(dataset.indices)
        base = getattr(dataset.dataset, "targets",
               getattr(dataset.dataset, "labels", None))
        labels = np.array(base)[idxs] if base is not None else np.array([dataset.dataset[i][1] for i in idxs])
    else:
        idxs = np.arange(n)
        base = getattr(dataset, "targets", getattr(dataset, "labels", None))
        labels = np.array(base) if base is not None else np.array([dataset[i][1] for i in idxs])

    classes = np.unique(labels)
    if min_classes > len(classes):
        raise ValueError(f"min_classes={min_classes} > #classes={len(classes)}")

    # Build per-class pools
    pools = {c: list(idxs[labels == c]) for c in classes}
    for c in pools:
        rng.shuffle(pools[c])

    dict_users = {u: [] for u in range(num_users)}

    # --- SEED PHASE: ensure >= min_classes per client ---
    for u in range(num_users):
        avail = [c for c in classes if len(pools[c]) > 0]
        chosen = rng.choice(avail, size=min(min_classes, len(avail)), replace=False)
        for c in chosen:
            take = min(seed_per_class, len(pools[c]))
            dict_users[u].extend(pools[c][:take])
            del pools[c][:take]

    # --- FILL PHASE: balanced size ---
    # Flatten remaining pool
    remaining = [idx for p in pools.values() for idx in p]
    rng.shuffle(remaining)

    for u in range(num_users):
        need = target_per_client - len(dict_users[u])
        if need > 0:
            dict_users[u].extend(remaining[:need])
            remaining = remaining[need:]

    # Final shuffle & convert to arrays
    for u in range(num_users):
        dict_users[u] = np.array(rng.permutation(dict_users[u]), dtype='int64')

    return dict_users


# Dataset subset for user-specific data
class DatasetSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = int(self.indices[idx])
        return self.dataset[data_idx]
#
# # Data preparation
# def prepare_data(args):
#     if args.dataset in ['CIFAR10', 'CIFAR100']:
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ]) if args.dataset == 'CIFAR10' else transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#         ])
#         dataset_class = datasets.CIFAR10 if args.dataset == 'CIFAR10' else datasets.CIFAR100
#     elif args.dataset == 'MNIST':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         dataset_class = datasets.MNIST
#     else:
#         raise ValueError(f"Unsupported dataset: {args.dataset}")
#
#     train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=transform)
#     test_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=transform)
#
#     return train_dataset, test_dataset

def prepare_data(args):
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]) if args.dataset == 'CIFAR10' else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset_class = datasets.CIFAR10 if args.dataset == 'CIFAR10' else datasets.CIFAR100
        train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=transform)
        test_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=transform)

    elif args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    elif args.dataset == 'HAM10000':
        train_dataset, val_dataset, test_dataset = get_ham_data()
        return train_dataset, val_dataset, test_dataset

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Default return path (CIFAR/MNIST)
    return train_dataset, test_dataset
# from torchvision import transforms, datasets
#
# def prepare_data(args):
#     if args.dataset in ['CIFAR10', 'CIFAR100']:
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ]) if args.dataset == 'CIFAR10' else transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#         ])
#         dataset_class = datasets.CIFAR10 if args.dataset == 'CIFAR10' else datasets.CIFAR100
#         train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=transform)
#         test_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=transform)
#         return train_dataset, test_dataset  # ✅ Return for CIFAR
#
#     elif args.dataset == 'MNIST':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
#         test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
#         return train_dataset, test_dataset  # ✅ Return for MNIST
#
#     elif args.dataset == 'HAM10000':
#         # ✅ Resize to 28x28 and normalize like CIFAR10
#         transform = transforms.Compose([
#             transforms.Resize((28, 28)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ])
#         train_dataset, val_dataset, test_dataset = get_ham_data(args.data_dir, transform)
#         return train_dataset, val_dataset, test_dataset  # ✅ Return 3 splits
#
#     else:
#         raise ValueError(f"Unsupported dataset: {args.dataset}")

# Validate model performance on validation set
def validate(model, val_loader, criterion, device=torch.device('cuda:0')):
    model = model.to(device)
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Early stopping mechanism to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, validation_loss, model):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0

def upload_cost(state_dict):
    total_memory_bytes = 0
    for key, value in state_dict.items():
        total_memory_bytes += value.numel() * value.element_size()
    total_memory_in_mb = total_memory_bytes / (1024 ** 2)
    return total_memory_in_mb

def number_parameters(model):
    n_ele = 0
    for _, para in model.named_parameters():
        n_ele += para.numel()
    return n_ele

def count_zeros_in_state_dict(state_dict):
    total_zeros = 0
    for key, value in state_dict.items():
        zero_count = (value == 0).sum().item()
        print(f"Parameter: {key}, Zero Count: {zero_count}")
        total_zeros += zero_count
    print(f"Total zeros in state_dict: {total_zeros}")
    return total_zeros

# Setup device based on device type
def setup_device(device_type: str):
    if device_type == "GPU":
        return select_free_gpu(max_memory_usage=0.5)
    return torch.device("cpu")

# Select the GPU with the least memory usage
def select_free_gpu(max_memory_usage=0.5):
    devices = GPUtil.getGPUs()
    available_gpus = [i for i in range(len(devices)) if devices[i].memoryUtil < max_memory_usage]
    if len(available_gpus) == 0:
        print(f"No available GPU with memory usage < {max_memory_usage}")
        selected_gpu = sorted(devices, key=lambda x: (x.memoryUtil, x.load))[0].id
    else:
        available_gpu_loads = [devices[i].load for i in available_gpus]
        min_load = min(available_gpu_loads)
        selected_gpu = available_gpus[available_gpu_loads.index(min_load)]

    print(f"Selected GPU: {selected_gpu} (memory = {devices[selected_gpu].memoryUtil * 100}%, load  {devices[selected_gpu].load * 100}%)")
    return torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")
