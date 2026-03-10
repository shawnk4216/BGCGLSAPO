import torch
import torch.nn as nn
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests

def get_token_gradients(model, inputs_dict, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the input token coordinates.
    """
    input_ids = inputs_dict["input_ids"][0]
    
    # Dynamically get the text embedding layer
    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight
    
    # 1. Create a one-hot representation of the text tokens we want to optimize
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    
    # 2. Multiply one-hot by embedding weights
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # 3. Get embeddings for the full sequence and stitch our differentiable graph into it
    embeds = embed_layer(input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :], 
            input_embeds, 
            embeds[:, input_slice.stop:, :]
        ], 
        dim=1
    )
    
    # 4. Prepare kwargs for the forward pass 
    # (Stripping input_ids to force the model to use our patched inputs_embeds)
    forward_kwargs = {k: v for k, v in inputs_dict.items() if k != "input_ids"}
    forward_kwargs["inputs_embeds"] = full_embeds
    
    # 5. Forward pass
    logits = model(**forward_kwargs).logits
    
    # 6. Calculate NLL loss on the target tokens
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
    loss.backward()
    
    # 7. Return the gradients of the one-hot vectors
    return one_hot.grad.clone()

def find_slice(full_sequence, subsequence):
    """Helper to find the slice of a subsequence within a larger 1D tensor."""
    seq_len = len(full_sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if torch.equal(full_sequence[i : i + sub_len], subsequence):
            return slice(i, i + sub_len)
    raise ValueError("Subsequence not found in the full sequence.")

# ==========================================
# 1. Initialization
# ==========================================
model_id = "google/gemma-3-12b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# ==========================================
# 2. Data Preparation
# ==========================================
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# The prompt we want to optimize
prompt_text = "Describe this image in detail."

# The target output we want to force the model to generate (or evaluate NLL against)
target_text = "The image is a close-up shot of a vibrant garden scene focusing on a cluster of pink cosmos flowers and a busy bumblebee."

# For GCG, the model needs the context AND the target stitched together 
# so we can backpropagate the loss of the target tokens to the input tokens.
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": target_text}]
    }
]

inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=False, # False because we already appended the assistant's response
    tokenize=True,
    return_dict=True, 
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

# ==========================================
# 3. Locate Slices for GCG
# ==========================================
input_ids_1d = inputs["input_ids"][0]

# Encode the raw strings (without special tokens) so we can find them in the final template
prompt_ids = torch.tensor(processor.tokenizer.encode(prompt_text, add_special_tokens=False), device=model.device)
target_ids = torch.tensor(processor.tokenizer.encode(target_text, add_special_tokens=False), device=model.device)

# Find where the prompt and target live inside the templated input_ids
input_slice = find_slice(input_ids_1d, prompt_ids)
target_slice = find_slice(input_ids_1d, target_ids)

# The loss is computed by shifting the logits by 1. 
# Logit at index `i-1` predicts the token at index `i`.
loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)

# ==========================================
# 4. Run Optimization Step
# ==========================================
print(f"Optimizing prompt slice: {input_slice} | Target slice: {target_slice}")

# We must clear inference_mode since we need gradients
with torch.enable_grad():
    # Freeze model parameters to save memory; we only want gradients on the one-hot input
    for param in model.parameters():
        param.requires_grad = False
        
    grads = get_token_gradients(model, inputs, input_slice, target_slice, loss_slice)

print("Gradient matrix shape:", grads.shape)
# Shape will be [len(prompt_ids), vocab_size]

# Example: Get the most salient token replacement for the first word in the prompt
top_k_indices = torch.topk(-grads[0], k=5).indices
suggested_tokens = processor.tokenizer.convert_ids_to_tokens(top_k_indices)

print(f"\nOriginal first token: '{processor.tokenizer.decode(prompt_ids[0])}'")
print(f"Top 5 GCG replacements to decrease NLL: {suggested_tokens}")