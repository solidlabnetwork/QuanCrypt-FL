import torch
import torchvision
import torch.nn as nn
import inversefed
import os

def gia(model, dataset, dict_users, dm, ds, device, model_name="ResNet18", save_dir="reconstruction_results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.manual_seed(42)

    user_indices = list(dict_users[0])
    img, label = dataset[user_indices[0]]
    ground_truth = img.to(device).unsqueeze(0)
    labels = torch.as_tensor([label], device=device)

    config = dict(signed=True,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim='adam',
                  restarts=8,
                  max_iterations=24_000,
                  total_variation=1e-1,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')

    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)

    model.zero_grad()
    target_loss = nn.CrossEntropyLoss()(model(ground_truth), labels)
    input_gradients = torch.autograd.grad(target_loss, model.parameters())
    input_gradients = [grad.detach() for grad in input_gradients]

    if not isinstance(input_gradients, list):
        input_gradients = [input_gradients]

    output, stats = rec_machine.reconstruct(input_gradients, labels, img_shape=(3, 32, 32))

    ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
    ground_truth_save_path = os.path.join(save_dir, f'QuantPHE_ground_truth_{model_name}_CIFAR10.png')
    torchvision.utils.save_image(ground_truth_denormalized, ground_truth_save_path)

    output_denormalized = torch.clamp(output * ds + dm, 0, 1)
    recon_image_save_path = os.path.join(save_dir, f'QuantPHE_reconstructed_{model_name}_CIFAR10.png')
    torchvision.utils.save_image(output_denormalized, recon_image_save_path)

    print(f"Saved ground truth image at: {ground_truth_save_path}")
    print(f"Saved reconstructed image at: {recon_image_save_path}")
