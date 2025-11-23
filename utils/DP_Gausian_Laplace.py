import math
from scipy import optimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import os
import numpy as np
import random
import pickle
from scipy import stats

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_on_client(client_model, dataloader, epochs=1, lr=0.01, weight_decay=0.001, device='cpu'):
    optimizer = optim.Adam(client_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    client_model.train()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return client_model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class DatasetSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        return self.dataset[data_idx]

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        if self.train:
            self.data = []
            self.labels = []
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

def cifar_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def compute_center_and_range(model, device='cpu'):
    C, R = [], []
    for param in model.parameters():
        param_data = param.data.to(device)
        range_ = 0.5 * (torch.max(param_data) - torch.min(param_data))
        center = (torch.max(param_data) + torch.min(param_data)) / 2
        C.append(center)
        R.append(range_)
    R = torch.tensor(R, dtype=torch.float32).to(device)
    C = torch.tensor(C, dtype=torch.float32).to(device)

    return C, R

def compute_sigma_agm(epsilon, delta, sensitivity):
    def phi(t):
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    def B_plus(v):
        return phi(math.sqrt(epsilon * v)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (v + 2)))

    def B_minus(u):
        return phi(-math.sqrt(epsilon * u)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (u + 2)))

    delta_0 = phi(0) - math.exp(epsilon) * phi(-math.sqrt(2 * epsilon))

    if delta >= delta_0:
        v_star = optimize.brentq(lambda v: B_plus(v) - delta, 0, 100)
        alpha = math.sqrt(1 + v_star / 2) - math.sqrt(v_star / 2)
    else:
        u_star = optimize.brentq(lambda u: B_minus(u) - delta, 0, 100)
        alpha = math.sqrt(1 + u_star / 2) + math.sqrt(u_star / 2)

    return alpha * sensitivity / math.sqrt(2 * epsilon)

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
        if (self.best_score is None) or (score > self.best_score + self.delta):
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

def compute_laplace_noise(epsilon, sensitivity):
    L = sensitivity / epsilon
    return stats.laplace(scale=L)