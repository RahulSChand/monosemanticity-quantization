import torch
import os
#! I need quant the main model
from transformer_lens import HookedTransformer

from transformer_lens import HookedTransformer
import torch

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


model = HookedTransformer.from_pretrained("gelu-2l").to(torch.float32).to("cuda:0")

# a = torch.load("something.bin")
# print(type(a))
# model.load_state_dict(torch.load("something.bin"))
#model.load_state_dict(torch.load("something.bin"))

for name, param in model.named_parameters():
    if "block" in name and "W" in name:
        param.data = fixed_fake_quantize(param.data, num_bits = 4)

torch.save(model.state_dict(), "quant_4_f.bin")


# def print_size_of_model(model):
#     torch.save(model.state_dict(), "temp.p")
#     print('Size (MB):', os.path.getsize("temp.p")/1e6)
#     os.remove('temp.p')

# def fake_quantize(w: torch.Tensor, num_bits: int = 8, symmetric: bool = True) -> torch.Tensor:
#     """
#     Simulate quantization of weights while keeping them in float32.
    
#     Args:
#         w: Input weights as torch.Tensor
#         num_bits: Number of bits to quantize to
#         symmetric: If True, use symmetric quantization around 0
    
#     Returns:
#         Quantized weights as float32 tensor
#     """

#     with torch.no_grad():
#         if symmetric:
#             n_levels = 2**(num_bits - 1) - 1  # One bit for sign
#             scale = torch.max(torch.abs(w))
#             min_val = -scale
#             max_val = scale
#         else:
#             n_levels = 2**num_bits - 1
#             min_val = torch.min(w)
#             max_val = torch.max(w)
        
#         scale = (max_val - min_val) / n_levels
        
#         # Clip weights to min/max range
#         w_clipped = torch.clamp(w, min_val, max_val)
        
#         # Quantize
#         w_quantized = torch.round((w_clipped - min_val) / scale) * scale + min_val
        
#         return w_quantized


# model = HookedTransformer.from_pretrained("gelu-2l").to(torch.float32).to("cuda:0")

# # model_dynamic_quantized = torch.ao.quantization.quantize_dynamic(
# #     model, dtype=torch.qint8
# # )

# for name, param in model.named_parameters():





# # print_size_of_model(model)
# # print_size_of_model(model_dynamic_quantized)

# # for name, param in model_dynamic_quantized.named_parameters():
# #     print(f"Layer: {name}, Shape: {param.shape}, Dtype: {param.dtype}")
