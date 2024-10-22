# %%
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

inputs = tokenizer.encode(
    prompt_data[0], return_tensors="pt", add_special_tokens=True
).to(device)

matrices = []
activated_features = get_active_features(prompt_data, tokenizer, model, device)
for layer in tqdm(range(len(activated_features) - 1)):
    grad_matrix = compute_gradient_matrix(
        inputs,
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
    draw_bipartite_graph(i, threshold=3)
