import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import random
import os
import json

# ==========================================
# 1. Core GCG Functions (From your snippet)
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

def find_slice(full_sequence, subsequence):
    """Helper to find the slice of a subsequence within a larger 1D tensor."""
    seq_len = len(full_sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if torch.equal(full_sequence[i : i + sub_len], subsequence):
            return slice(i, i + sub_len)
    raise ValueError("Subsequence not found in the full sequence.")

# ==========================================
# 2. Mediator LLM Function
# ==========================================
def query_mediator_llm(processor, model, original_prompt, saliency_report, token_nll_dict):
    """
    Uses the Mediator LLM (in this case, the same model) to rewrite the prompt 
    based on the GCG gradient saliency and loss score.
    """
    mediator_system_prompt = (
        "You are an expert Prompt Engineer. Your task is to optimize a task prompt based on gradient-based token saliency. "
        "You will be given the original prompt, its Negative Log-Likelihood (NLL) score, and a list of tokens "
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
    # 3.1 Initialize Model & Processor
    model_id = "google/gemma-3-12b-it" # Or a smaller model for testing
    
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 3.2 Load Dataset (GSM8K/SVAMP parquet files)
    print(f"Loading dataset from {train_parquet_path}...")
    df = pd.read_parquet(train_parquet_path)
    # Assuming columns 'question' and 'answer' based on HuggingFace standard
    dataset = df.to_dict('records')
    
    # 3.3 Initialize Seed Prompt
    current_prompt = "Solve the following math word problem step by step and provide the final answer."
    print(f"\n[Seed Prompt]: {current_prompt}")
    
    for iteration in range(iterations):
        print(f"\n{'='*40}\nIteration {iteration + 1}/{iterations}\n{'='*40}")
        
        # Sample a small batch for this iteration to get gradient signals
        batch = random.sample(dataset, batch_size)


        saliency_reports = []
        representative_nll_dict = {}

        for idx, item in enumerate(batch):
            question_text = item['question']
            target_text = item['answer']
            
            # Combine the current prompt and the question
            full_user_text = f"{current_prompt}\n\nQuestion: {question_text}"
            
            messages = [
                {"role": "user", "content": [{"type": "text", "text": full_user_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
            ]
            
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_ids_1d = inputs["input_ids"][0]
            
            # Find slices (Using text without special tokens to locate)
            prompt_ids = torch.tensor(processor.tokenizer.encode(current_prompt, add_special_tokens=False), device=model.device)
            target_ids = torch.tensor(processor.tokenizer.encode(target_text, add_special_tokens=False), device=model.device)
            
            try:
                input_slice = find_slice(input_ids_1d, prompt_ids)
                target_slice = find_slice(input_ids_1d, target_ids)
            except ValueError:
                # If exact slice fails due to tokenization boundaries, skip this sample
                continue 
                
            loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
            
            # Run GCG optimization step
            with torch.enable_grad():
                for param in model.parameters():
                    param.requires_grad = False
                grads, scalar_loss, token_losses = get_token_gradients(model, inputs, input_slice, target_slice, loss_slice)

            # Generate saliency report for the Mediator
            if not representative_nll_dict:
                target_tokens = processor.tokenizer.convert_ids_to_tokens(target_ids)
                representative_nll_dict = {
                    f"{i:03d}_{tok}": round(float(loss), 4) 
                    for i, (tok, loss) in enumerate(zip(target_tokens, token_losses))
                }
                
            report = f"Sample {idx+1} (Avg Loss: {scalar_loss:.4f}):\n"
            for i in range(len(prompt_ids)):
                # Top 3 gradient replacements for token 'i'
                top_k_indices = torch.topk(-grads[i], k=3).indices
                suggested_tokens = processor.tokenizer.convert_ids_to_tokens(top_k_indices)
                orig_token = processor.tokenizer.decode(prompt_ids[i])
                report += f"  - Token '{orig_token}' -> GCG produced: {suggested_tokens}\n"
            
            saliency_reports.append(report)
            
        if not saliency_reports:
            print("Failed to find token slices in this batch. Retrying...")
            continue
            
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
    # Ensure you have 'pyarrow' or 'fastparquet' installed to read the parquet files.
    # pip install pyarrow pandas torch transformers accelerate
    
    train_file = "train-00000-of-00001.parquet"
    if os.path.exists(train_file):
        final_prompt = optimize_prompt_pipeline(train_file, iterations=5, batch_size=2)
    else:
        print("Please ensure the parquet files are in the working directory.")
