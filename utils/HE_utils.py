import tenseal as ts
from tqdm import tqdm
import time
import torch

def encrypt_weights(model_update, context, encryption_times):
    encrypted_update = {}
    for k, v in tqdm(model_update.items(), desc="Encrypting"):
        start_time = time.time()
        encrypted_weight = ts.ckks_vector(context, v.detach().cpu().numpy().flatten())
        encryption_times.append(time.time() - start_time)
        encrypted_update[k] = encrypted_weight
    return encrypted_update

def decrypt_weights(encrypted_update, context, model, decryption_times):
    decrypted_update = {}
    for name, encrypted_weight in tqdm(encrypted_update.items(), desc="Decrypting"):
        start_time = time.time()
        decrypted_weight = encrypted_weight.decrypt()
        decryption_times.append(time.time() - start_time)
        decrypted_update[name] = torch.tensor(decrypted_weight).view_as(model.state_dict()[name]).to(model.state_dict()[name].device)
    return decrypted_update

def aggregate_encrypted_updates(encrypted_updates, context):
    aggregated_update = {}
    for k in encrypted_updates[0].keys():
        encrypted_sum = encrypted_updates[0][k]
        for update in encrypted_updates[1:]:
            encrypted_sum += update[k]
        encrypted_avg = encrypted_sum * (1.0 / len(encrypted_updates))
        aggregated_update[k] = encrypted_avg
    return aggregated_update
