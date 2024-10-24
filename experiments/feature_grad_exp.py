# %%
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.sae_funcs import (
    compute_gradient_matrix,
    get_active_features,
)

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

prompt_data["prompts"] = [
    " ".join(sentence.split()[:-1]) for sentence in prompt_data["sentences"]
]

# %%
with open(
    "/root/feature-circuit-discovery/datasets/ioi/ioi_test_100_2.json", "w"
) as file:
    json.dump(prompt_data, file)


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

matrices = []

activated_features = get_active_features(tokenized_prompts, model, device, threshold=0.5)

for i in range(len(activated_features)):
    print()
    print(activated_features[i])
    print("length:",len(activated_features[i]))

# %%
for layer in tqdm(range(len(activated_features) - 1)):
    grad_matrix = compute_gradient_matrix(
        tokenized_prompts,
        layer,
        layer + 1,
        activated_features[layer],
        activated_features[layer + 1],
        model,
        verbose=True,
    )
    matrices.append(grad_matrix)

# %%
from feature_circuit_discovery.graphs import draw_bipartite_graph

for i in matrices:
    draw_bipartite_graph(i, threshold=0)

# %%
