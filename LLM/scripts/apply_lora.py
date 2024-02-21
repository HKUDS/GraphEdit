import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
 
 
def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    model_name_or_path = './vicuna_7b_v1.5_16k/'
    output_path = './vicuna_7b_pubmed/'
    lora_path = './Pubmed_model/'

    apply_lora(model_name_or_path, output_path, lora_path)