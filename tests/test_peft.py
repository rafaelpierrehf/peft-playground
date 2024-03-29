import pytest
from llmin.core.minimizer import Minimizer

@pytest.fixture(scope = "session")
def minimizer():

        mini = Minimizer(
            target_model_id = "mistralai/Mistral-7B-v0.1",
            cache_dir = "/tmp/models"
        )
        return mini

def test_get_lora_model(minimizer):
    
    base_model = minimizer.compress()
    adapters = minimizer.get_lora_adapters(base_model = base_model)
    merged = adapters.merge_and_unload()
    assert merged


