import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
import random
import numpy as np
import re
from torch.nn.functional import cosine_similarity
import torch._dynamo 
torch._dynamo.config.disable = True
# ++++++++++++++++++
from tqdm import tqdm

def extract_ground_truth(answer_str):
    if "#### " in answer_str:
        return answer_str.split("#### ")[-1].replace(",", "").strip()
    return None

def extract_model_prediction(generated_text):
    match = re.search(r'<answer>(.*?)</answer>', generated_text)
    if match:
        # Extract the number and strip commas
        number_str = re.sub(r'[^\d.-]', '', match.group(1))
        return number_str
    return None

def evaluate_prompt(model, tokenizer, dataset_split, system_prompt, batch_size=16):
    correct = 0
    total = len(dataset_split)

    for i in range(0, total, batch_size):
        batch = dataset_split[i : i + batch_size]
        questions = batch["question"]
        ground_truths = [extract_ground_truth(ans) for ans in batch["answer"]]

        batch_messages = [
            [
                {"role": "system", "content": f"{system_prompt}\n \n Wrap your final numerical answer in XML tags like this: <answer>9</answer>\n"},
                {"role": "user", "content": q}
            ]
            for q in questions
        ]

        formatted_prompts = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            text=formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for response, truth in zip(responses, ground_truths):
            prediction = extract_model_prediction(response)
            if prediction == truth:
                correct += 1

    return correct / total
# ++++++++++++++++++

"""
Citation: Zou et al. (2023)
This is a tweaked version of the original GCG attack code
"""
def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    one_hot: torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    loss: float
        The loss from the forward pass
    loss_per_token:
        The cross entropy loss with respect to each individual token
    """

    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(
        one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype
        )
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = embed_layer(input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1).contiguous()
    
    custom_mask = torch.ones_like(input_ids, device=model.device).unsqueeze(0)

    logits = model(inputs_embeds=full_embeds, attention_mask=custom_mask, use_cache=False).logits
    targets = input_ids[target_slice]
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss_per_token = loss_fn(logits[0,loss_slice,:], targets)
    loss = torch.mean(loss_per_token)

    loss.backward()
    one_hot_gradients = one_hot.grad.clone()
    model.zero_grad()

    return one_hot_gradients, loss.item()


def prompt_similarity(model, tokenizer, prompt_a, prompt_b):
    """Returns cosine similarity between mean token embeddings of two prompts."""
    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        ids_a = tokenizer(text=prompt_a, return_tensors="pt")["input_ids"].to(model.device)
        ids_b = tokenizer(text=prompt_b, return_tensors="pt")["input_ids"].to(model.device)
        emb_a = embed_layer(ids_a).mean(dim=1)
        emb_b = embed_layer(ids_b).mean(dim=1)
    return cosine_similarity(emb_a, emb_b).item()
# NOTE: threshold lowered to 0.6 to compensate for the coarseness of embedding-based
# similarity — lexical variation between synonymous prompts scores lower here than
# with hidden-state similarity, so a tighter threshold rejects too many good candidates.


def prompt_manager(model, tokenizer, prompt_selection, original_prompt, saliency_report, start_prompt, rejected_prompts=None):
    """
    Takes the model, tokenizer, and model inputs in order to generate the new iterated prompt
    """
    system_prompts = {
        "baseline": (
            "You are an expert Prompt Engineer specializing in iterative refinement and prompt compression. "
            "Your ONLY job is to make targeted improvements to the existing task prompt. "
            "You will be given the PREVIOUS prompt, and you must optimize it such that it remains "
            "highly likely to produce a correct answer while using as few tokens (words) as possible.\n\n"
            "RULES:\n"
            "1. Preserve the core instructional meaning and tone, but make it more concise.\n"
            "2. Make sure the prompt is coherent and interpretable.\n"
            "3. Remove filler words, redundant phrasing, and unnecessary conversational text.\n"
            "4. The new prompt should ideally be SHORTER than the previous prompt, but accuracy is still the primary goal.\n"
            "Output ONLY the new (refined) prompt text. No explanations."
            " Do not even output a label for the new prompt (such as 'Here is the new prompt:')"
        ),
        "gcg": (
            "You are an expert Prompt Engineer specializing in compression. "
            "You will be given the PREVIOUS prompt and a list of mathematically important keywords identified by a gradient-based algorithm. "
            "These keywords represent the tokens most critical to the model's ability to solve math problems correctly.\n\n"
            "YOUR GOAL: Produce a SHORTER version of the prompt that preserves its full meaning and effectiveness.\n\n"
            "CRITICAL RULES:\n"
            "1. THE OUTPUT MUST BE STRICTLY SHORTER (fewer words) THAN THE PREVIOUS PROMPT. If you cannot shorten it, remove at least one redundant word.\n"
            "2. THE PROMPT MUST REMAIN AN IMPERATIVE COMMAND instructing an AI to solve math problems. NEVER turn it into a question.\n"
            "3. Use the keyword suggestions as a guide to which concepts are essential — keep those concepts, cut everything else.\n"
            "4. Remove filler words, redundant phrases, and unnecessary qualifiers. Prefer concise, dense phrasing.\n"
            "5. CRITICAL ANCHOR: The rewritten prompt MUST still clearly instruct an AI to solve mathematical word problems correctly. Accuracy must not be sacrificed.\n"
            "6. If a suggested keyword is already captured by a shorter synonym, prefer the shorter form.\n"
            "Output ONLY the new (compressed) prompt text. No explanations."
        ),
    }

    if rejected_prompts:
        recent = rejected_prompts[-5:]
        rejected_block = "PREVIOUSLY REJECTED PROMPTS (these failed to improve — do NOT repeat them):\n"
        for entry in recent:
            rejected_block += f"  - Prompt: '{entry['prompt']}'\n    Reason: {entry['reason']}\n"
        rejected_block += "\n"
    else:
        rejected_block = ""

    user_prompts = {
        "baseline": (
            f"For reference, this is the prompt we first started with (ensure the core instruction is the same): {start_prompt}\n"
            f"PREVIOUS prompt:\n'{original_prompt}'\n\n"
            f"{rejected_block}"
            "Make targeted improvements to increase accuracy AND reduce length. Output ONLY the new compressed prompt."
        ),
        "gcg": (
            f"PREVIOUS prompt:\n'{original_prompt}'\n\n"
            f"Mathematically Important Keywords (identified by gradient analysis — these tokens matter most):\n{saliency_report}\n\n"
            f"{rejected_block}"
            "Compress the prompt: rewrite it as a shorter imperative command for an AI solving math problems. "
            "Use the keyword list to identify what is essential — preserve those concepts, cut everything else. "
            "The output MUST be shorter than the previous prompt. Output ONLY the new compressed prompt."
        ),
    }

    user_prompt = user_prompts[prompt_selection]
    system_prompt = system_prompts[prompt_selection]

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content":  user_prompt}
    ]
    
    # Step 1: Render the template as a string
    formatted_chat = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Step 2: Tokenize using the explicit text= argument
    tokenized_chat = tokenizer(
        text=formatted_chat,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **tokenized_chat, 
            max_new_tokens=128, 
            temperature=0.6, 
            do_sample=True
        )

    new_prompt = tokenizer.decode(outputs[0][len(tokenized_chat["input_ids"][0]):], skip_special_tokens=True)
    return new_prompt

def clean_tokens(model, tokenizer, unclean_string):
    system_prompt = (
        "You are an expert linguistic data cleaner. "
        "You will be given a raw list of word-replacement suggestions generated by a mathematical algorithm. "
        "Most of the suggestions are garbage (HTML tags, foreign words, fragments, punctuation, special tokens). "
        "For each original word, extract the top 3 to 5 VALID, COHERENT English synonyms from the messy list. "
        "CRITICAL RULE: If a word's suggestion list contains ZERO valid English synonyms, output an empty list []. Do not force a bad word."
        "Ignore any words that are punctuation, fragments, or gibberish. "
        "Output ONLY the cleaned list in this format:\n"
        "'original_word' --> ['clean1', 'clean2', 'clean3']\n"
        "Do not include any other text or explanations."
    )

    messages = [
        {"role": "system", "content" : system_prompt}, 
        {"role": "user", "content": f"RAW SUGGESTIONS:\n{unclean_string}"}
    ]
    
    # Step 1: Render as string
    formatted_chat = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Step 2: Tokenize using text=
    tokenized_chat = tokenizer(
        text=formatted_chat, 
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **tokenized_chat, 
            max_new_tokens=256, 
            temperature=0.5, 
            do_sample=True
        )

    clean_suggestions = tokenizer.decode(outputs[0][len(tokenized_chat["input_ids"][0]):], skip_special_tokens=True)
    return clean_suggestions

# def optimize_prompt(model, tokenizer, dataset, iterations, prompt_selection):
    
#     train = list(dataset["train"])
#     test = list(dataset["test"])

#     for param in model.parameters():
#         param.requires_grad = False

    
#     task_prompt = "Please answer the following questions as accurately as possible."
#     original_prompt = task_prompt

#     batch_size = 25
#     for i in range(iterations):
#         unclean_token_suggestions = ""

        
#         task_prompt_tensor = tokenizer(task_prompt, add_special_tokens=False, return_tensors = "pt" )["input_ids"].to(model.device)
#         task_len = task_prompt_tensor.shape[1]

#         avg_gradients = torch.zeros(task_len, model.get_input_embeddings().weight.shape[0], device=model.device)

#         for l in range(batch_size):

#             if "gcg" not in prompt_selection:
#                 continue

#             example = random.choice(train)
#             question = example["question"]
#             answer = example["answer"]

            
#             question_tensor = tokenizer(question, add_special_tokens=False, return_tensors = "pt")["input_ids"].to(model.device)
#             answer_tensor = tokenizer(answer, add_special_tokens=False, return_tensors = "pt")["input_ids"].to(model.device)

            
#             question_len = question_tensor.shape[1]
#             answer_len = answer_tensor.shape[1]

#             input_tensor = torch.cat((task_prompt_tensor, question_tensor, answer_tensor), dim=1)[0]

#             task_slice = slice(0, task_len)
#             target_slice = slice(task_len + question_len, task_len + question_len + answer_len)
#             loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)

#             with torch.enable_grad():
#                 gradients, loss = token_gradients(model, input_tensor, task_slice, target_slice, loss_slice)
            
#             avg_gradients += gradients

#         k = 10

#         if "gcg" in prompt_selection:
#             avg_gradients = avg_gradients / batch_size
#             top_k_indices = torch.topk(-avg_gradients, k=32)[1]
#             for j, vector in enumerate(top_k_indices):

              
#                 raw_original = tokenizer.decode(input_tensor[j].item())
#                 clean_original = re.sub(r"[^A-Za-z]", "", raw_original)
                
#                 if len(clean_original) <= 1:
#                     continue

#                 decoded_words = tokenizer.batch_decode([[v] for v in vector], skip_special_tokens=True)
#                 token_list = []

#                 for token in decoded_words:
#                     clean_token = re.sub(r"[^A-Za-z]", "", token)
#                     if len(clean_token) > 1:
#                         token_list.append(clean_token)
                    
#                 unclean_token_suggestions += f"'{raw_original.strip()}'   -->   {token_list[:k]} \n"
            
#             token_suggestions = clean_tokens(model, tokenizer, unclean_token_suggestions)
#             print(f"Cleaned Suggestions:\n{token_suggestions}\n")
#         else:
#             token_suggestions = ""

#         new_prompt = prompt_manager(model, tokenizer, prompt_selection, task_prompt, token_suggestions, original_prompt)
#         SIMILARITY_THRESHOLD = 0.76

#         sim = prompt_similarity(model, tokenizer, task_prompt, new_prompt)
#         if sim < SIMILARITY_THRESHOLD:
#             print(f"Rejected update (similarity={sim:.3f}), keeping previous prompt.")
#         else:
#             task_prompt = new_prompt
#             print(f"Current Iterated Prompt: {task_prompt}")
    
#     return task_prompt

# ++++++++++++++++++
def optimize_prompt(
    model,
    tokenizer,
    dataset,
    iterations,
    prompt_selection,
    eval_split_name="test",
    eval_subset_size=100,
    eval_every=1
):
    train = list(dataset["train"])
    eval_split = dataset[eval_split_name]

    if eval_subset_size is not None:
        eval_split = eval_split.select(range(min(eval_subset_size, len(eval_split))))

    for param in model.parameters():
        param.requires_grad = False

    task_prompt = "Please answer the following questions as accurately as possible. Think step-by-step and provide reasoning for your solutions. Double-check your answers and list your thoughts clearly."
    original_prompt = task_prompt
    batch_size = 25
    similarity_threshold = 0.2

    history = []
    rejected_prompts = []  # only accuracy-drop rejections, capped at 5 in prompt_manager
    accuracy_change_threshold = 0.06

    initial_accuracy = evaluate_prompt(
        model=model,
        tokenizer=tokenizer,
        dataset_split=eval_split,
        system_prompt=task_prompt,
        batch_size=16
    )

    initial_token_count = tokenizer(text=task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]

    history.append({
        "variation": prompt_selection,
        "iteration": 0,
        "accepted_update": True,
        "similarity": 1.0,
        "accuracy": initial_accuracy,
        "token_count": initial_token_count,
        "prompt": task_prompt
    })

    print(f"[{prompt_selection}] Iteration 0 | Accuracy: {initial_accuracy:.4f} | Tokens: {initial_token_count}")
    print(f"[{prompt_selection}] Prompt: {task_prompt}\n")

    for i in range(1, iterations + 1):
        unclean_token_suggestions = ""

        task_prompt_tensor = tokenizer(
            text=task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].to(model.device)

        task_len = task_prompt_tensor.shape[1]
        avg_gradients = torch.zeros(
            task_len,
            model.get_input_embeddings().weight.shape[0],
            device=model.device
        )

        if "gcg" in prompt_selection:
            for _ in range(batch_size):
                example = random.choice(train)
                question = example["question"]
                # FIX 1: Use only the final numeric answer as the GCG target,
                # NOT the full chain-of-thought. Targeting the full CoT string
                # (hundreds of tokens the model can't reproduce verbatim) produces
                # near-random gradient signal and prevents convergence.
                answer = extract_ground_truth(example["answer"])
                if answer is None:
                    continue

                question_tensor = tokenizer(
                    text=question,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"].to(model.device)

                answer_tensor = tokenizer(
                    text=answer,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"].to(model.device)

                question_len = question_tensor.shape[1]
                answer_len = answer_tensor.shape[1]

                input_tensor = torch.cat(
                    (task_prompt_tensor, question_tensor, answer_tensor),
                    dim=1
                )[0]

                task_slice = slice(0, task_len)
                target_slice = slice(
                    task_len + question_len,
                    task_len + question_len + answer_len
                )
                loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)

                with torch.enable_grad():
                    gradients, _ = token_gradients(
                        model, input_tensor, task_slice, target_slice, loss_slice
                    )

                avg_gradients += gradients
                
                del input_tensor, gradients, question_tensor, answer_tensor
                torch.cuda.empty_cache()
                
            avg_gradients = avg_gradients / batch_size
            top_k_indices = torch.topk(-avg_gradients, k=32)[1]
            k = 10

            for j, vector in enumerate(top_k_indices):
                raw_original = tokenizer.decode(task_prompt_tensor[0][j].item())
                clean_original = re.sub(r"[^A-Za-z]", "", raw_original)

                if len(clean_original) <= 1:
                    continue

                decoded_words = tokenizer.batch_decode(
                    [[v] for v in vector],
                    skip_special_tokens=True
                )

                token_list = []
                for token in decoded_words:
                    clean_token = re.sub(r"[^A-Za-z]", "", token)
                    if len(clean_token) > 1:
                        token_list.append(clean_token)

                unclean_token_suggestions += f"'{raw_original.strip()}' --> {token_list[:k]}\n"

            token_suggestions = clean_tokens(model, tokenizer, unclean_token_suggestions)
            print(f"Cleaned Suggestions:\n{token_suggestions}\n")
        else:
            token_suggestions = ""

        candidate_prompt = prompt_manager(
            model,
            tokenizer,
            prompt_selection,
            task_prompt,
            token_suggestions,
            original_prompt,
            rejected_prompts=rejected_prompts
        )

        sim = prompt_similarity(model, tokenizer, task_prompt, candidate_prompt)

        # --- Gate 1: similarity filter ---
        if sim < similarity_threshold:
            accepted_update = False
            print(f"[{prompt_selection}] Rejected at iteration {i} (similarity={sim:.3f} < {similarity_threshold})")
            accuracy = history[-1]["accuracy"]
            token_count = history[-1]["token_count"]
        else:
            candidate_accuracy = evaluate_prompt(
                model=model,
                tokenizer=tokenizer,
                dataset_split=eval_split,
                system_prompt=candidate_prompt,
                batch_size=16
            )
            candidate_token_count = tokenizer(text=candidate_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
            current_accuracy = history[-1]["accuracy"]
            current_token_count = history[-1]["token_count"]
            accuracy_change = candidate_accuracy - current_accuracy  # signed

            if prompt_selection == "gcg":
                # --- GCG gate: minimize tokens, with accuracy as a floor ---
                # Reject if accuracy drops more than the threshold.
                # Accept if tokens are reduced (or equal) and accuracy is maintained.
                # Reject if candidate is longer — that defeats the purpose.
                if accuracy_change < -accuracy_change_threshold:
                    accepted_update = False
                    token_count = current_token_count
                    accuracy = current_accuracy
                    reason = (
                        f"accuracy dropped: {current_accuracy:.4f} → {candidate_accuracy:.4f} "
                        f"(drop={-accuracy_change:.4f} > {accuracy_change_threshold})"
                    )
                    rejected_prompts.append({"prompt": candidate_prompt, "reason": reason})
                    print(f"[{prompt_selection}] Rejected at iteration {i} ({reason})")
                elif candidate_token_count >= current_token_count:
                    accepted_update = False
                    token_count = current_token_count
                    accuracy = current_accuracy
                    reason = (
                        f"not shorter: {current_token_count} → {candidate_token_count} tokens"
                    )
                    rejected_prompts.append({"prompt": candidate_prompt, "reason": reason})
                    print(f"[{prompt_selection}] Rejected at iteration {i} ({reason})")
                else:
                    task_prompt = candidate_prompt
                    accepted_update = True
                    token_count = candidate_token_count
                    accuracy = candidate_accuracy
                    print(
                        f"[{prompt_selection}] Accepted update at iteration {i} "
                        f"(similarity={sim:.3f}, tokens={current_token_count}→{token_count}, accuracy={accuracy:.4f})"
                    )
            else:
                # --- Baseline gate: original accuracy-maximizing logic ---
                if accuracy_change < -accuracy_change_threshold:
                    accepted_update = False
                    accuracy = current_accuracy
                    token_count = current_token_count
                    if accuracy_change < 0:
                        reason = (
                            f"accuracy dropped: {current_accuracy:.4f} → {candidate_accuracy:.4f} "
                            f"(drop={-accuracy_change:.4f} > {accuracy_change_threshold})"
                        )
                        rejected_prompts.append({"prompt": candidate_prompt, "reason": reason})
                        print(f"[{prompt_selection}] Rejected at iteration {i} ({reason})")
                    else:
                        print(f"[{prompt_selection}] Rejected at iteration {i} (accuracy spike too large: {current_accuracy:.4f} → {candidate_accuracy:.4f})")
                else:
                    task_prompt = candidate_prompt
                    accepted_update = True
                    accuracy = candidate_accuracy
                    token_count = candidate_token_count
                    print(f"[{prompt_selection}] Accepted update at iteration {i} (similarity={sim:.3f}, accuracy={accuracy:.4f})")

        if i % eval_every == 0:
            history.append({
                "variation": prompt_selection,
                "iteration": i,
                "accepted_update": accepted_update,
                "similarity": sim,
                "accuracy": accuracy,
                "token_count": token_count,
                "prompt": task_prompt
            })

            print(f"[{prompt_selection}] Iteration {i} | Accuracy: {accuracy:.4f} | Tokens: {token_count}")
            print(f"[{prompt_selection}] Prompt: {task_prompt}\n")

    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{prompt_selection}_iteration_metrics.csv", index=False)

    if prompt_selection == "gcg":
        best_row = history_df.loc[history_df["token_count"].idxmin()]
        print(
            f"[{prompt_selection}] Best iteration: {int(best_row['iteration'])} | "
            f"Tokens: {int(best_row['token_count'])} | Accuracy: {best_row['accuracy']:.4f}"
        )
    else:
        best_row = history_df.loc[history_df["accuracy"].idxmax()]
        print(
            f"[{prompt_selection}] Best iteration: {int(best_row['iteration'])} | "
            f"Accuracy: {best_row['accuracy']:.4f}"
        )

    return task_prompt, history_df
# ++++++++++++++++++


# random.seed(42)
# torch.manual_seed(42)
# device = torch.device("cuda:3")

# variations = ["baseline", "gcg"]
# dataset = load_dataset("gsm8k", "main")



# model_id = "Qwen/Qwen2.5-14B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, device_map=device, torch_dtype=torch.bfloat16,
# ).eval()

# final_prompts= {}

# for variation in variations:
#     final_prompts[variation] = optimize_prompt(model, tokenizer, dataset, 30, variation)

# for variation, prompt in final_prompts.items():
#     print(f"{variation}: {prompt} \n")

# ++++++++++++++++++
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda:3")

variations = ["baseline", "gcg"]
dataset = load_dataset("gsm8k", "main")

model_id = "google/gemma-3-12b-it"
tokenizer = AutoProcessor.from_pretrained(model_id, padding_side="left")

if tokenizer.tokenizer.pad_token is None:
    tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=torch.bfloat16,
).eval()

final_prompts = {}
all_histories = []

for variation in variations:
    final_prompt, history_df = optimize_prompt(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        iterations=30,
        prompt_selection=variation,
        eval_split_name="test",
        eval_subset_size=100,   # increase later to 250/full test if runtime is acceptable
        eval_every=1
    )

    final_prompts[variation] = final_prompt
    all_histories.append(history_df)

all_history_df = pd.concat(all_histories, ignore_index=True)
all_history_df.to_csv("all_iteration_metrics.csv", index=False)

print("\n=== FINAL PROMPTS ===")
for variation, prompt in final_prompts.items():
    print(f"{variation}: {prompt}\n")
# ++++++++++++++++++