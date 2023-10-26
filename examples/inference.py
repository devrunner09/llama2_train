# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import torch
from transformers import LlamaTokenizer

# from llama_recipes.inference.safety_utils import get_safety_checker
# from llama_recipes.inference.model_utils import load_model, load_peft_model


from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig

# Function to load the main model for text generation
def load_model(model_name, quantization):
    print(f"Loading model from: {model_name}")
    print(f"Quantization: {quantization}")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def main(
    model_name,
    peft_model: str="/home/khuongnd6/TuyenDT5/ViLLM/26k_finetuned_r8",
    quantization: bool=True,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    prompt_file: str="/home/khuongnd6/TuyenDT5/llama-recipes/examples/alpaca_prompt.txt",
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.1, # [optional] The value used to modulate the next token probabilities.
    top_k: int=30, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = "\n".join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    print("====User Prompt====")
    print(user_prompt + "\n")
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    print("Loaded base model")

    if peft_model:
        model = load_peft_model(model, peft_model)
    print("Loaded PEFT model")

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    print("Loaded tokenizer model")

    model.resize_token_embeddings(model.config.vocab_size + 1) 
    batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")

    batch = {k: v.to("cuda") for k, v in batch.items()}
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text)

                

if __name__ == "__main__":
    model_name = "/home/khuongnd6/TuyenDT5/ViLLM/finetuned_models"
    main(model_name)
