import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import numpy as np
import re
from torch.nn.functional import cosine_similarity

# ++++++++++++++++++
from tqdm import tqdm

def extract_ground_truth(answer_str):
    if "#### " in answer_str:
        return answer_str.split("#### ")[-1].replace(",", "").strip()
    return None

def extract_model_prediction(generated_text):
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', generated_text)
    if numbers:
        return numbers[-1].replace(",", "")
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
                {"role": "system", "content": system_prompt},
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
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
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
        ids_a = tokenizer(prompt_a, return_tensors="pt")["input_ids"].to(model.device)
        ids_b = tokenizer(prompt_b, return_tensors="pt")["input_ids"].to(model.device)
        emb_a = embed_layer(ids_a).mean(dim=1)
        emb_b = embed_layer(ids_b).mean(dim=1)
    return cosine_similarity(emb_a, emb_b).item()


def prompt_manager(model, tokenizer, prompt_selection, original_prompt, saliency_report, start_prompt):
    """
    Takes the model, tokenizer, and model inputs in order to generate the new iterated prompt
    """
    system_prompts = {
        "baseline": (
            "You are an expert Prompt Engineer specializing in iterative refinement. "
            "Your ONLY job is to make targeted improvements to the existing task prompt. "
            "You will be given the PREVIOUS prompt, and you have to optimize the prompt "
            "such that it is more likely to produce a correct answer \n\n"
            "RULES:\n"
            "1. Preserve the original meaning and tone.\n"
            "2. Make sure the prompt is coherent and interpretable.\n"
            "Output ONLY the new (refined) prompt text. No explanations."
            " Do not even output a label for the new prompt (such as 'Here is the new prompt:')"
        ),
        "gcg": (
            "You are an expert Prompt Engineer specializing in iterative refinement. "
            "You will be given the PREVIOUS prompt and a list of mathematically suggested keyword replacements. "
            "Because these suggestions are generated by a raw mathematical algorithm, doing direct 1-to-1 word swaps will often destroy the grammar of the sentence.\n\n"
            "CRITICAL RULES:\n"
            "1. THE PROMPT MUST REMAIN AN IMPERATIVE COMMAND. It must explicitly instruct an AI to answer questions or solve math problems. NEVER turn the prompt into a question itself.\n"
            "2. DO NOT act like a strict search-and-replace tool. Do not blindly swap words.\n"
            "3. Review the suggested words. Identify 1 or 2 of the most powerful, logical English words from the list.\n"
            "4. REWRITE the previous prompt into a highly professional, grammatically perfect COMMAND that naturally incorporates your chosen keywords. If a keyword is a noun like 'business' or 'luxury', weave it into the context (e.g., '...even if the math problem involves luxury businesses').\n"
            "5. If a suggested replacement makes no grammatical or logical sense, IGNORE IT entirely.\n"
            # FIX 3: Added rule 6 to prevent semantic drift. Without this, the LLM
            # accepts GCG suggestions that are plausible English but destroy task
            # meaning (e.g. 'equations' -> 'movie', 'sitcom'), causing the prompt to
            # stop instructing math solving and accuracy to degrade.
            "6. CRITICAL ANCHOR: The rewritten prompt MUST still clearly instruct an AI to solve mathematical word problems or answer math questions correctly. Never let keyword suggestions pull the prompt away from this core math-solving purpose.\n"
            "Output ONLY the new (refined) prompt text. No explanations."
        ),
    
    }

    user_prompts = {
        "baseline": (
            f"For reference, this is the prompt we first started with (ensure the meaning is the same): {start_prompt}\n"
            f"PREVIOUS prompt:\n'{original_prompt}'\n\n"
            "Make significant, targeted improvements. Output ONLY the new prompt."
        ),
        "gcg": (
            f"PREVIOUS prompt:\n'{original_prompt}'\n\n"
            f"Mathematical Keyword Suggestions:\n{saliency_report}\n\n"
            "Rewrite the prompt as a highly professional imperative command instructing an AI to solve math problems. Use 1 or 2 valid suggestions as inspiration. Output ONLY the new prompt."
        ),

    }

    user_prompt = user_prompts[prompt_selection]
    system_prompt = system_prompts[prompt_selection]

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content":  user_prompt}
]
    tokenized_chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**tokenized_chat, max_new_tokens=128, disable_compile=True, temperature=0.6, do_sample=True)

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
    
    tokenized_chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**tokenized_chat, max_new_tokens=256, disable_compile=True, temperature=0.1, do_sample=True)

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

    task_prompt = "Please answer the following questions as accurately as possible."
    original_prompt = task_prompt
    batch_size = 25
    similarity_threshold = 0.76

    history = []

    initial_accuracy = evaluate_prompt(
        model=model,
        tokenizer=tokenizer,
        dataset_split=eval_split,
        system_prompt=task_prompt,
        batch_size=16
    )

    history.append({
        "variation": prompt_selection,
        "iteration": 0,
        "accepted_update": True,
        "similarity": 1.0,
        "accuracy": initial_accuracy,
        "prompt": task_prompt
    })

    print(f"[{prompt_selection}] Iteration 0 | Accuracy: {initial_accuracy:.4f}")
    print(f"[{prompt_selection}] Prompt: {task_prompt}\n")

    for i in range(1, iterations + 1):
        unclean_token_suggestions = ""

        task_prompt_tensor = tokenizer(
            task_prompt,
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
                    question,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"].to(model.device)

                answer_tensor = tokenizer(
                    answer,
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
            original_prompt
        )

        sim = prompt_similarity(model, tokenizer, task_prompt, candidate_prompt)
        accepted_update = sim >= similarity_threshold

        # FIX 2: Update task_prompt HERE, before the eval block, so that:
        #   (a) the accuracy is measured on the new (accepted) prompt, not the old one,
        #   (b) the NEXT iteration's task_prompt_tensor / avg_gradients are computed
        #       on the latest accepted prompt rather than one step behind.
        # Previously, task_prompt was updated here but task_prompt_tensor was built
        # at the top of the loop using the *pre-update* value, meaning GCG gradients
        # always lagged one iteration behind the prompt being evaluated.
        if accepted_update:
            task_prompt = candidate_prompt
            print(f"[{prompt_selection}] Accepted update at iteration {i} (similarity={sim:.3f})")
        else:
            print(f"[{prompt_selection}] Rejected update at iteration {i} (similarity={sim:.3f})")

        if i % eval_every == 0:
            accuracy = evaluate_prompt(
                model=model,
                tokenizer=tokenizer,
                dataset_split=eval_split,
                system_prompt=task_prompt,
                batch_size=16
            )

            history.append({
                "variation": prompt_selection,
                "iteration": i,
                "accepted_update": accepted_update,
                "similarity": sim,
                "accuracy": accuracy,
                "prompt": task_prompt
            })

            print(f"[{prompt_selection}] Iteration {i} | Accuracy: {accuracy:.4f}")
            print(f"[{prompt_selection}] Prompt: {task_prompt}\n")

    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{prompt_selection}_iteration_metrics.csv", index=False)

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

variations = ["gcg", "baseline"]
dataset = load_dataset("gsm8k", "main")

model_id = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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