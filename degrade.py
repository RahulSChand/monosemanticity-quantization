import torch
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset

# Get data first
model = HookedTransformer.from_pretrained("gelu-2l")  # Temporary model just for tokenizer
data = load_dataset("NeelNanda/c4-code-20k", split="train")
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(42)
tokens = tokenized_data["tokens"][:256]

print(tokens.shape)

del model  # Delete the temporary model

# Test normal model
print("Testing normal model...")
normal_model = HookedTransformer.from_pretrained("gelu-2l").to(torch.bfloat16).cuda()
with torch.no_grad():
    normal_loss = normal_model(tokens, return_type="loss")
print(f"Normal model loss: {normal_loss:.4f}")
del normal_model  # Free up memory
torch.cuda.empty_cache()

# Test quantized model
print("\nTesting quantized model...")
quant_model = HookedTransformer.from_pretrained("gelu-2l", init_weights=True).to(torch.bfloat16).cuda()
# for name, param in quant_model.named_parameters():
#     if "block" in name and "W" in name:
#         torch.nn.init.kaiming_uniform_(param.data) 

quant_model.load_state_dict(torch.load("quant_8.bin"))
with torch.no_grad():
    quant_loss = quant_model(tokens, return_type="loss")
print(f"4-bit quantized model loss: {quant_loss:.4f}")
print(f"Loss difference: {quant_loss - normal_loss:.4f}")
print(f"Relative degradation: {(quant_loss - normal_loss)/normal_loss:.2%}")