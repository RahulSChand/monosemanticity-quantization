from datasets import load_dataset,load_from_disk
import torch
import einops
from transformer_lens import HookedTransformer

data = load_from_disk("c4_code_tokenized_2b.hf")
data.set_format(type="torch", columns=["tokens"])
all_tokens = data["tokens"]
print(all_tokens.shape)



model = HookedTransformer.from_pretrained("gelu-2l").to(torch.float32).to("cuda:0")

all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]

print(all_tokens_reshaped.numel())

# torch.save(all_tokens_reshaped, "c4_code_2b_tokens_reshaped.pt")

# else:
#     # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
#     all_tokens = torch.load("/workspace/data/c4_code_2b_tokens_reshaped.pt")
#     all_tokens = shuffle_data(all_tokens)