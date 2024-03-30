import os
from transformers import AutoModelForCausalLM
from huggingface_hub import list_repo_files, snapshot_download
from typing import Tuple, LiteralString
from peft.tuners.tuners_utils import replicate_layers
from peft.peft_model import PeftModel
from peft import AutoPeftModelForCausalLM
from peft.config import PeftConfig
from peft.mixed_model import PeftMixedModel
from peft.tuners.lora.config import LoraConfig
import copy
from llmin.utils import get_linear_module_names
from llmin.svd import decompose_delta_weight
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import torch
from collections import OrderedDict


class Minimizer:

    def __init__(
        self,
        target_model_id: str,
        compression_factor: float = 0.5,
        cache_dir: str = None
    ):
        self.target_model_id = target_model_id
        self.compression_factor = compression_factor
        self.cache_dir = cache_dir
        self.target_model_path = self._download_shards()

    def _download_shards(self) -> Tuple[LiteralString, bool]:

        ignore_patterns = ["*.safetensors"]
        local_path = snapshot_download(
            repo_id=self.target_model_id,
            cache_dir=self.cache_dir,
            ignore_patterns=ignore_patterns,
            max_workers = 12
        )

        return local_path
    
    def compress(self) -> AutoModelForCausalLM:

        target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_path,
            device_map="cpu"
        )
        base_model_num_layers = int(len(target_model.model.layers) * self.compression_factor)
        original_layers = [0, base_model_num_layers]
        additional_layers = [[base_model_num_layers - 1, base_model_num_layers]] * base_model_num_layers
        layer_map = [original_layers]
        layer_map.extend(additional_layers)

        base_model = copy.deepcopy(target_model)
        replicate_layers(model = base_model,layer_map = layer_map)
        del target_model
        
        return base_model
    
    def get_base_lora(
        self,
        linear_module_names,
        rank = 32,
        alpha = 1.0,
        output_dir: str = "/tmp/peft/"
    ) -> dict:

        lora_config = LoraConfig(
            lora_alpha=alpha, # Setting the alpha to the to decomposition rank value (instead of alpha value used) seems to give better performance. Further testing would be needed to understand what is the optimal alpha value to use
            lora_dropout=0,
            r=rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= list(set([self.get_module_peft_name(e) for e in linear_module_names])),
        )

        model = self.compress()
        peft_model = get_peft_model(model, lora_config)

        return peft_model

    def get_lora_adapters(
        self,
        base_model,
        rank = 32,
        alpha = 1.0,
        path: str = "/tmp/peft/adapter_model.bin"
    ):
        lora_adapters = None
        target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_path,
            device_map = "cpu"
        )

        linear_module_names = get_linear_module_names(target_model)
        lora_adapters = self.get_base_lora(linear_module_names=linear_module_names)

        if not os.path.exists(path) or not lora_adapters:
            base_state_dict = base_model.state_dict()
            target_state_dict = target_model.state_dict()

            lora_adapters = OrderedDict()

            for module in tqdm(linear_module_names):
                target_tensor = target_state_dict[module + ".weight"]
                base_tensor = base_state_dict[module + ".weight"]

                lora_A, lora_B = decompose_delta_weight(
                    new_weight = target_tensor,
                    base_weight = base_tensor,
                    alpha = alpha,
                    reduced_rank = rank
                )

                lora_adapters[f"base_model.model.{module}.lora_A.weight"] = lora_A
                lora_adapters[f"base_model.model.{module}.lora_B.weight"] = lora_B

            torch.save(lora_adapters, "/tmp/peft/adapter_model.bin")

        #for key in lora_adapters.keys():
            #lora_adapters[key] = lora_adapters[key].to('cpu').contiguous()
        
        return lora_adapters
    
    def get_peft_model(self, base_model, lora_config):

        auto_peft_model = PeftModel.from_pretrained(
            model = base_model,
            config = lora_config
        )
        del base_model
        return auto_peft_model

    def get_module_peft_name(self, module_name):
        return module_name.split('.')[-1]
    


