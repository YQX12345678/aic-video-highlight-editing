"""Video LLM SFT for highlight localization and cropping."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def build_lora_model(base_model_name: str):
    """Load video LLM and apply LoRA for SFT."""
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def prepare_sft_data(video_features, instruction: str, highlights: list):
    """Create instruction tuning data mapping (features, instruction) -> timestamps."""
    target_str = " ".join([f"<{s}-{e}>" for s, e in highlights])
    return {"instruction": instruction, "output": target_str}
