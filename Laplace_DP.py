from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision import datasets, transforms
import pandas as pd
import time
from utils.Nets import *
from utils.DP_Gausian_Laplace import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
set_seed(42)

model_name = 'ResNet18'
n_clients = 50
n_local_epochs = 1
num_rounds = 300
learning_rate = 0.001
weight_decay = 0.0001
epsilon = 3.0
lambda_param = 1.0  # Smothing Param

transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_dir = './data/cifar-10-batches-py'
train_dataset = CIFAR10Dataset(data_dir=data_dir, train=True, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
test_dataset = CIFAR10Dataset(data_dir=data_dir, train=False, transform=transform)

dict_users = cifar_iid(train_dataset, n_clients)
client_dataloaders = [DataLoader(Subset(train_dataset, list(dict_users[i])), batch_size=64, shuffle=True) for i in range(n_clients)]
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

initial_model = ResNet18().to(device)
global_model = ResNet18().to(device)
global_model.load_state_dict(initial_model.state_dict())

accuracy_list = []
train_accuracy_list = []
results = []

start_time = time.time()

C_list = []
R_list = []

# Define checkpoint directory
checkpoint_dir = 'Checkpoint'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_Laplace_eps_{epsilon}_1.pth')
checkpoint_path2 = os.path.join(checkpoint_dir, f'checkpoint_Laplace_eps_{epsilon}_2.pth')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    start_round = checkpoint['round']
    accuracy_list = checkpoint['accuracy_list']
    train_accuracy_list = checkpoint['train_accuracy_list']
    C_list = checkpoint['C_list']
    R_list = checkpoint['R_list']
    results = checkpoint['results']
else:
    start_round = 0
    accuracy_list = []
    train_accuracy_list = []
    results = []
    C_list = []
    R_list = []
early_stopping = EarlyStopping(patience=5)
best_accuracy = 0.0

for round in range(start_round, num_rounds):
    C, R = compute_center_and_range(global_model, device=device)
    C_list.append(C)
    R_list.append(R)

    global_state_dict = global_model.state_dict()
    c_r_keys = [k for k in global_state_dict.keys() if "weight" in k or "bias" in k]
    c_r_mapping = {k: i for i, k in enumerate(c_r_keys)}
    local_models = []

    for i in range(n_clients):
        client_model = ResNet18().to(device)
        client_model.load_state_dict(global_state_dict)

        # Train the client model
        train_on_client(client_model, client_dataloaders[i], epochs=n_local_epochs, lr=learning_rate,
                        weight_decay=weight_decay, device=device)

        perturbed_state_dict = client_model.state_dict()
        for key in c_r_keys:
            param = perturbed_state_dict[key]
            param_data = param.data
            num_elements = param_data.numel()
            c = C[c_r_mapping[key]].item()
            r = R[c_r_mapping[key]].item()
            sensitivity = 2 * r

            laplace = compute_laplace_noise(epsilon, sensitivity)
            noise = torch.tensor(laplace.rvs(size=param_data.shape), dtype=torch.float32).to(device)
            param.data.add_(noise)
        local_models.append(perturbed_state_dict)

    for key in global_state_dict.keys():
        if global_state_dict[key].dtype == torch.long:
            global_state_dict[key] = (1 - lambda_param) * global_state_dict[key] + lambda_param * \
                                     torch.stack([local_model[key].float() for local_model in local_models], 0).mean(0).long()
        else:
            global_state_dict[key] = (1 - lambda_param) * global_state_dict[key] + lambda_param * \
                                     torch.stack([local_model[key] for local_model in local_models], 0).mean(0)

    global_model.load_state_dict(global_state_dict)

    global_model.eval()
    test_loss, test_accuracy = validate(global_model, test_dataloader, nn.CrossEntropyLoss())
    accuracy_list.append(test_accuracy)
    print(f'Round {round + 1}, Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}')

    train_loss, train_accuracy = validate(global_model, client_dataloaders[0], nn.CrossEntropyLoss())
    train_accuracy_list.append(train_accuracy)
    print(f'Round {round + 1}, Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.4f}')

    val_loss, val_accuracy = validate(global_model, val_loader, nn.CrossEntropyLoss())
    print(f'Round {round + 1}, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
    print('--------------------------------------------------------')

    results.append([round + 1, train_accuracy, test_accuracy, val_accuracy])

    early_stopping(val_loss, global_model)
    if early_stopping.early_stop:
        print("Early stopping triggered. Reloading the best model and continuing training.")
        global_model.load_state_dict(early_stopping.best_model)
        early_stopping.early_stop = False
        early_stopping.counter = 0

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(global_model.state_dict(), checkpoint_path2)
        print(f"Saving the best model with validation accuracy: {val_accuracy:.2f}%")

    # Save checkpoint
    torch.save({
        'round': round + 1,
        'model_state_dict': global_model.state_dict(),
        'accuracy_list': accuracy_list,
        'train_accuracy_list': train_accuracy_list,
        'C_list': C_list,
        'R_list': R_list,
        'results': results
    }, checkpoint_path)

# Define output directory and filename for Excel file
output_dir = 'result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
filename = f"{output_dir}/laplace_DP_FL_{model_name}_{epsilon}_{lambda_param}_{n_clients}.xlsx"

# Create DataFrame and save to Excel
results_df = pd.DataFrame(results, columns=["Round", "Train Accuracy", "Test Accuracy", "Validation Accuracy"])
results_df.to_excel(filename, index=False, engine='xlsxwriter')

end_time = time.time()
total_training_time = end_time - start_time
print(f"Total Training Time: {total_training_time:.2f} seconds")
