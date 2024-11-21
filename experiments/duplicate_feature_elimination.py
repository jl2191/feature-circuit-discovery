
# %%    
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib.widgets import Slider
from transformers import AutoModelForCausalLM, AutoTokenizer
from feature_circuit_discovery.data import canonical_sae_filenames
from feature_circuit_discovery.sae_funcs import (
    compute_gradient_matrix,
    get_active_features,
    describe_features,
    load_sae,
    identify_duplicate_features,
    get_active_features_in_layer
)

# %%
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto",
)

device = model.device
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# %%
with open(
    "/tmp/feature-circuit-discovery/datasets/ioi/ioi_test_100.json", "rb"
) as file:
    prompt_data = json.load(file)

prompt_data["prompts"] = [
    " ".join(sentence.split()[:-1]) for sentence in prompt_data["sentences"]
]

# %%

tokenized_prompts = (
    tokenizer(
        prompt_data["prompts"][:100],
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    .data["input_ids"]
    .to(device)
)
dataset = TensorDataset(tokenized_prompts)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

upstream_layer = 15
downstream_layer = 16
upstream_features = torch.tensor(get_active_features_in_layer(tokenized_prompts, model, upstream_layer))
downstream_features = torch.tensor(get_active_features_in_layer(tokenized_prompts, model, downstream_layer))
# %%

identify_duplicate_features(model, dataloader, upstream_layer, downstream_layer, upstream_features, downstream_features, verbose=True)

# %%
