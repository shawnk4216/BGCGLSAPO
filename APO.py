import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM
import random
import os
import json

# ==========================================
# 1. Core GCG Functions
# ==========================================
def get_token_gradients(model, inputs_dict, input_slice, target_slice, loss_slice):
    """Computes gradients of the loss w.r.t the input token coordinates."""
    input_ids = inputs_dict["input_ids"][0]
    
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
    forward_kwargs = {k: v for k, v in inputs_dict.items() if k != "input_ids"}
    forward_kwargs["inputs_embeds"] = full_embeds
    
    # 5. Forward pass
    logits = model(**forward_kwargs).logits
    
    targets = input_ids[target_slice]

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fn(logits[0, loss_slice, :], targets)

    # We still need a scalar to backpropagate the gradients to our one_hot vector
    scalar_loss = token_losses.mean()
    scalar_loss.backward()

    # Return gradients, the scalar loss, AND the detached token-level losses
    return one_hot.grad.clone(), scalar_loss.item(), token_losses.detach()

# ==========================================
# 2. Mediator LLM Function
# ==========================================
def query_mediator_llm(processor, model, original_prompt, saliency_report, token_nll_dict):
    """
    Uses the Mediator LLM to rewrite the prompt based on the GCG gradient saliency 
    and target NLL score.
    """
    mediator_system_prompt = (
        "You are an expert Prompt Engineer. Your task is to optimize a task prompt based on gradient-based token saliency. "
        "You will be given the original prompt, its per-token Negative Log-Likelihood (NLL) score, and a list of tokens "
        "produced by the Greedy Coordinate Gradient algorithm that contribute the most towards lowering the NLL of the target output.\n"
        "Rewrite the prompt so that it incorporates these semantic suggestions to improve reasoning, but ensure the new prompt "
        "remains human-readable, coherent, and grammatically correct. It is not necessary to include all suggestions or feedback as long as the prompt has converged and been optimized. Output ONLY the new prompt text."
    )
    
    # Format the NLL dictionary nicely for the LLM
    formatted_nll_dict = json.dumps(token_nll_dict, indent=2)

    mediator_user_prompt = (
        f"Original Prompt: '{original_prompt}'\n\n"
        f"Input Token Saliency and Suggested Replacements:\n{saliency_report}\n"
        f"Output Target Per-Token NLL Dictionary (Higher = Model struggled more):\n{formatted_nll_dict}\n\n"
        "Based on these insights, provide the revised optimal prompt."
    )
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": mediator_system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": mediator_user_prompt}]}
    ]
    
    # NOTE: Do NOT cast to bfloat16 here. Token IDs must remain integers.
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        
    # Extract just the generated text
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    new_prompt = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return new_prompt

# ==========================================
# 3. Main APO Optimization Pipeline
# ==========================================
def optimize_prompt_pipeline(train_parquet_path, iterations=5, batch_size=3):
    model_id = "google/gemma-3-12b-it" 
    
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Freeze model parameters globally (saves time inside the loop)
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Loading dataset from {train_parquet_path}...")
    df = pd.read_parquet(train_parquet_path)
    dataset = df.to_dict('records')
    
    current_prompt = "Solve the following math word problem step by step and provide the final answer."
    print(f"\n[Seed Prompt]: {current_prompt}")
    
    for iteration in range(iterations):
        print(f"\n{'='*40}\nIteration {iteration + 1}/{iterations}\n{'='*40}")
        
        batch = random.sample(dataset, batch_size)

        saliency_reports = []
        representative_nll_dict = {}

        for idx, item in enumerate(batch):
            question_text = item['question']
            target_text = item['answer']
            
            # --- FORCED TOKEN CONCATENATION ---
            # 1. Setup Base Tokens (handle BOS dynamically)
            bos_id = processor.tokenizer.bos_token_id
            bos_tensor = torch.tensor([[bos_id]], device=model.device) if bos_id is not None else torch.empty((1,0), dtype=torch.long, device=model.device)
            
            # 2. Tokenize components separately to freeze their boundaries
            prompt_ids_2d = processor.tokenizer.encode(current_prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            question_ids_2d = processor.tokenizer.encode(f"\n\nQuestion: {question_text}\nAnswer: ", add_special_tokens=False, return_tensors="pt").to(model.device)
            target_ids_2d = processor.tokenizer.encode(target_text, add_special_tokens=False, return_tensors="pt").to(model.device)

            # 3. Calculate exact lengths
            bos_len = bos_tensor.shape[1]
            prompt_len = prompt_ids_2d.shape[1]
            question_len = question_ids_2d.shape[1]
            target_len = target_ids_2d.shape[1]

            # 4. Define mathematical slices
            prompt_start = bos_len
            input_slice = slice(prompt_start, prompt_start + prompt_len)

            target_start = bos_len + prompt_len + question_len
            target_slice = slice(target_start, target_start + target_len)
            loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)

            # 5. Stitch tensors together
            input_ids = torch.cat([bos_tensor, prompt_ids_2d, question_ids_2d, target_ids_2d], dim=1)
            attention_mask = torch.ones_like(input_ids)
            
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            
            # Extract 1D IDs for downstream decoding
            prompt_ids = prompt_ids_2d.squeeze(0)
            target_ids = target_ids_2d.squeeze(0)
            
            # Run GCG optimization step
            with torch.enable_grad():
                grads, scalar_loss, token_losses = get_token_gradients(model, inputs, input_slice, target_slice, loss_slice)

            # Generate NLL dictionary (capturing the first sample in the batch as representative)
            if not representative_nll_dict:
                target_tokens = processor.tokenizer.convert_ids_to_tokens(target_ids)
                representative_nll_dict = {
                    f"{i:03d}_{tok}": round(float(loss), 4) 
                    for i, (tok, loss) in enumerate(zip(target_tokens, token_losses))
                }
                
            # Generate saliency report for this sample
            report = f"Sample {idx+1} (Avg Loss: {scalar_loss:.4f}):\n"
            for i in range(len(prompt_ids)):
                top_k_indices = torch.topk(-grads[i], k=3).indices
                suggested_tokens = processor.tokenizer.convert_ids_to_tokens(top_k_indices)
                orig_token = processor.tokenizer.decode(prompt_ids[i])
                report += f"  - Token '{orig_token}' -> GCG produced: {suggested_tokens}\n"
            
            saliency_reports.append(report)
            
        # Combine all reports from the batch
        combined_report = "\n".join(saliency_reports)
        print("Passing GCG Saliency and Target NLL Dictionary to Mediator LLM...")
        
        # 3.4 Mediator LLM generates the new prompt using the NLL Dictionary
        new_prompt = query_mediator_llm(processor, model, current_prompt, combined_report, representative_nll_dict)
        print(f"\n[Mediator Revised Prompt]:\n{new_prompt}")
        
        current_prompt = new_prompt
        
    print("\nOptimization Complete.")
    print(f"Final Optimized Prompt: {current_prompt}")
    return current_prompt

if __name__ == "__main__":
    train_file = "train-00000-of-00001.parquet"
    if os.path.exists(train_file):
        final_prompt = optimize_prompt_pipeline(train_file, iterations=5, batch_size=2)
    else:
        print("Please ensure the parquet files are in the working directory.")
