import torch
from transformer_lens import HookedTransformer
def fake_quantize(w: torch.Tensor, num_bits: int = 8, symmetric: bool = True) -> torch.Tensor:
    """
    Simulate quantization of weights while keeping them in float32.
    
    Args:
        w: Input weights as torch.Tensor
        num_bits: Number of bits to quantize to
        symmetric: If True, use symmetric quantization around 0
    
    Returns:
        Quantized weights as float32 tensor
    """

    with torch.no_grad():
        if symmetric:
            n_levels = 2**(num_bits - 1) - 1  # One bit for sign
            scale = torch.max(torch.abs(w))
            min_val = -scale
            max_val = scale
        else:
            n_levels = 2**num_bits - 1
            min_val = torch.min(w)
            max_val = torch.max(w)
        
        scale = (max_val - min_val) / n_levels
        # print(scale)
        
        # Clip weights to min/max range
        w_clipped = torch.clamp(w, min_val, max_val)
        
        # Quantize
        w_quantized = torch.round((w_clipped - min_val) / scale) * scale + min_val
        
        return w_quantized

def fixed_fake_quantize(w: torch.Tensor, num_bits: int = 8, symmetric: bool = True) -> torch.Tensor:
    with torch.no_grad():
        if symmetric:
            n_levels = 2**(num_bits - 1) - 1
            max_abs = torch.max(torch.abs(w))
            scale = max_abs / n_levels  # Divide by n_levels to preserve scale
            min_val = -max_abs
            max_val = max_abs
        else:
            n_levels = 2**num_bits - 1
            min_val = torch.min(w)
            max_val = torch.max(w)
            scale = (max_val - min_val) / n_levels
        
        w_clipped = torch.clamp(w, min_val, max_val)
        w_int = torch.round((w_clipped - min_val) / scale)
        w_quantized = w_int * scale + min_val
        
        return w_quantized


def analyze_quantization(w: torch.Tensor, num_bits: int = 4):
    """
    Analyze what happens during quantization compared to random noise
    """
    with torch.no_grad():
        # Get original stats
        orig_std = w.std().item()
        orig_mean = w.mean().item()
        orig_range = (w.min().item(), w.max().item())
        
        # Quantize
        w_quant = fixed_fake_quantize(w, num_bits=num_bits)
        quant_std = w_quant.std().item()
        quant_mean = w_quant.mean().item()
        
        # Add comparable noise
        noise = torch.randn_like(w) * orig_std
        noise_mean = noise.mean().item()
        noise_std = noise.std().item()
        
        # Calculate number of unique values
        unique_vals = torch.unique(w_quant).numel()
        
        # Calculate how many weights got clamped to min/max
        if num_bits == 4:
            n_levels = 2**(num_bits - 1) - 1
            scale = torch.max(torch.abs(w))
            clamp_count = torch.sum((w_quant == scale) | (w_quant == -scale)).item()
            clamp_percent = 100 * clamp_count / w.numel()
            
        print(f"Original - std: {orig_std:.4f}, mean: {orig_mean:.4f}, range: {orig_range}")
        print(f"Quantized - std: {quant_std:.4f}, mean: {quant_mean:.4f}")
        print(f"Noise - std: {noise_std:.4f}, mean: {noise_mean:.4f}")
        print(f"Unique quantized values: {unique_vals}")
        print(f"Percent weights clamped: {clamp_percent:.2f}%")
        
        return w_quant, noise

model = HookedTransformer.from_pretrained("gelu-2l").to(torch.float32).to("cuda:0")

# Test on one of your weight matrices
for name, param in model.named_parameters():
    if "block" in name and "W" in name:
        print(f"\nAnalyzing {name}:")
        w_quant, noise = analyze_quantization(param.data)
        break