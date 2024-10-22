#! Find the error w.r.t to the pre-trained model

#! Train the model myself
import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch

model = HookedTransformer.from_pretrained("gelu-2l").to(torch.float32).to("cuda:0")

for name, param in model.named_parameters():
    print(name, param.shape)

new_cfg = {
  "n_layers": 2,
  "d_model": 512,
  "d_mlp": 2048,
  "d_head": 64,
  "n_heads": 8,
  "n_ctx": 1024,
  "d_vocab": 48262,
  "normalization_type": None,
  "tokenizer_name": "NeelNanda/gpt-neox-tokenizer-digits",
  "act_fn": "gelu",
}
new_model = HookedTransformer(new_cfg)

print("-----------")

for name, param in new_model.named_parameters():
    print(name, param.shape)