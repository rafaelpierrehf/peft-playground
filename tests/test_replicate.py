from llmin.core.minimizer import Minimizer
import pytest
import tempfile
import shutil

@pytest.fixture(scope = "session")
def minimizer():

    mini = Minimizer(
        target_model_id = "mistralai/Mistral-7B-v0.1",
        cache_dir = "/tmp/models/"
    )
    return mini

def test_init(minimizer):

    assert minimizer.target_model_path

def test_compress(minimizer):

    minimizer.compress()

def test_lora(minimizer):

    lora = minimizer.get_lora_adapters()

