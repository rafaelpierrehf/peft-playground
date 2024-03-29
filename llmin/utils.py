import torch
from transformers import AutoModelForCausalLM
import glob
from safetensors import safe_open

def get_linear_embedding_layers(model_type):
    """
    returns the linear embedding layers needed for loras, dependent on the model arch
    """
    if model_type == "gpt_neox":
        return ["embed_in", "embed_out"]
    if model_type == "falcon":
        return ["word_embeddings", "lm_head"]
    return ["embed_tokens", "lm_head"]


def get_linear_module_names(model):
    
    cls = torch.nn.Linear
    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names.append(name)


    return names


def load_safetensors(target_model_path: str, framework="pt", device='cpu'):
    """
    Loads tensors from .safetensors files in the specified directory into a dictionary.

    Args:
    - directory (str): Path to the directory containing .safetensors files.
    - framework (str): The framework to use ('pt' for PyTorch, 'tf' for TensorFlow, etc.). Default is 'pt'.
    - device (str): The device to load the tensors on ('cpu', 'cuda', etc.). Default is 'cpu'.

    Returns:
    - dict: A dictionary containing all tensors from the .safetensors files.
    """
    tensors_dict = {}
    # Use glob to find all .safetensors files in the directory
    file_paths = glob.glob(f"{target_model_path}/*.safetensors")

    # Loop through each file and load its tensors into the dictionary
    for file_path in sorted(file_paths):
        with safe_open(file_path, framework=framework, device=device) as f:
            for k in f.keys():
                tensors_dict[k] = f.get_tensor(k)

    return tensors_dict