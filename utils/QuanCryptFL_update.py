import torch
from torch import nn
from utils.Prune import prune_model, set_weight_by_mask  # Import unified functions

class client_train(object):
    def __init__(self, args, device, lr, weight_decay, dataloader):
        self.args = args
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = dataloader

    def train(self, net, current_round, pruning_mask=None):
        net = net.to(self.device)
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        epoch_loss = []

        for epoch in range(self.args.n_local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # Pruning logic
            effective_round = self.args.effective_round
            target_round = self.args.target_round
            initial_pruning_rate = self.args.initial_pruning_rate
            target_pruning_rate = self.args.target_pruning_rate

            pruning_rate = max(0, (current_round - effective_round) /
                               (target_round - effective_round)) * \
                           (target_pruning_rate - initial_pruning_rate) + initial_pruning_rate

            if current_round >= self.args.effective_round:
                # Unified pruning for ResNet and other CNN models
                is_resnet = self.args.model_name == 'ResNet18'
                pruning_mask = prune_model(
                    model=net,
                    sparsity=pruning_rate,
                    is_resnet=is_resnet,
                    include_fc=True,
                    include_bias=False
                )
                set_weight_by_mask(model=net, mask=pruning_mask)
                count_zero_params(net)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), pruning_mask
def count_zero_params(model):
    total_params = 0
    total_zeros = 0

    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            total_zeros += torch.sum(param == 0).item()

    zero_ratio = 100.0 * total_zeros / total_params if total_params != 0 else 0
    print(f"Total Parameters: {total_params}")
    print(f"Zero Parameters: {total_zeros}")
    print(f"Sparsity: {zero_ratio:.2f}%")

    return total_params, total_zeros, zero_ratio
