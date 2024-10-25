# %%
import json
from pprint import pprint

import pandas as pd
from IPython.display import display
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.sae_funcs import (
    describe_features,
    get_active_features,
)

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto",
)

device = model.device


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")


# %%
prompt_data = [
    "what's your favourite color?",
    "who should be president?",
    "why don't you like me?",
]

# %%
with open(
    "/root/feature-circuit-discovery/datasets/ioi/ioi_test_100.json", "rb"
) as file:
    prompt_data = json.load(file)


# %%
tokenized_prompts = (
    tokenizer(
        prompt_data["prompts"][:15],
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    .data["input_ids"]
    .to(device)
)

activated_features = get_active_features(
    tokenized_prompts, model, device, threshold=0.7
)

# %%
pprint(activated_features)

# %%
# convert activated features from tensors to lists
activated_features = {
    layer: features.tolist() for layer, features in activated_features.items()
}
pprint(activated_features)

# %%
activated_feature_descriptions = describe_features(activated_features)

pprint(activated_feature_descriptions)

# %%
# what is the distribution of often to less often activated features?
# this will be a plot of activated feature id and frequency. might also help to color by
# layer

# %%

flattened_features = [
    {"layer": layer, "feature_id": feature["feature_id"]}
    for layer, features in activated_feature_descriptions.items()
    for feature in features
]

activated_features_df = pd.DataFrame(flattened_features)

display(activated_features_df)

# %%
