import torch

def quantize_tensor(x, num_bits=8):
    """
    Symmetric quantization of a tensor to any bit width (1 to 32).
    Returns:
        q_x: Quantized tensor (int32)
        scale: The scale factor used
    """
    if not (1 <= num_bits <= 32):
        raise ValueError(f"Invalid bit-width {num_bits}. Must be between 1 and 32.")

    # Compute symmetric quantization range
    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    # Compute scale based on absolute max
    max_val = torch.max(torch.abs(x)).item()
    if max_val == 0:
        scale = 1.0  # Prevent divide by zero
    else:
        scale = max_val / qmax

    # Quantize: scale and clamp
    q_x = torch.round(x / scale).clamp(qmin, qmax).to(torch.int32)

    return q_x, scale


def dequantize_tensor(scale, q_x):
    """
    Dequantize a quantized tensor using the scale.
    Supports both torch.Tensor and float inputs for q_x.
    """
    scale = float(scale)
    if isinstance(q_x, torch.Tensor):
        return q_x.float() * scale
    else:
        return torch.tensor(q_x, dtype=torch.float32) * scale

