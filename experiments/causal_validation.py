# %%
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from feature_circuit_discovery.sae_funcs import *
import matplotlib.pyplot as plt
# %%

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto",
)

device = model.device


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# %%
with open(
    "/root/feature-circuit-discovery/datasets/ioi/ioi_test_100.json", "rb"
) as file:
    prompt_data = json.load(file)

prompt_data["prompts"] = [
    " ".join(sentence.split()[:-1]) for sentence in prompt_data["sentences"]
]

# %%
#Here we are calculating the our method's "guess" for what the causal connection between two features will be.

tokenized_prompts = (
    tokenizer(
        prompt_data["prompts"][:10],
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    .data["input_ids"]
    .to(device)
)

upstream_layer = 18
downstream_layer = 20
upstream_features = get_active_features_in_layer(tokenized_prompts, model, 18)
downstream_features = get_active_features_in_layer(tokenized_prompts, model, 20)

grad_matrix = compute_gradient_matrix(
    tokenized_prompts,
    upstream_layer,
    downstream_layer,
    upstream_features,
    downstream_features,
    model,
    verbose=True,
)


# %%
print(downstream_features)
# %%
#Here we attempt to validate or prediction by seeing what kind of impact the ablation/reinforcement of certain feature activations may have.
upstream_sae = load_sae(canonical_sae_filenames[upstream_layer], device)
downstream_sae = load_sae(canonical_sae_filenames[downstream_layer], device)
scalars = [-1+i*0.1 for i in range(21)]
modified_feature_acts = []
for feature_idx in tqdm(upstream_features):
    row = []
    for scalar in scalars:
        logits, ds_acts = model_with_added_feature_vector(model, tokenized_prompts, upstream_sae, upstream_layer, feature_idx, scalar, downstream_layer)
        ds_feature_acts = downstream_sae.encode(ds_acts)
        row.append(ds_feature_acts[:, -1, downstream_features])
    modified_feature_acts.append(row)


# %%
print("number of upstream features:", len(upstream_features))
print("number of downstream features:", len(downstream_features))
print("number of samples:", len(tokenized_prompts))
print("number of scalars:", len(scalars))

# %%
stacked_scalars = [torch.stack(scalar_list, dim=0) for scalar_list in modified_feature_acts]

# Step 2: Stack all upstream features
# This will convert the list of 17 tensors (each of shape [21, 10, 15]) into a single tensor of shape [17, 21, 10, 15]
modified_feature_acts = torch.stack(stacked_scalars, dim=0).reshape(17,15, 21, 10)

# %%
print(modified_feature_acts.shape)
# %%
modified_feature_acts =modified_feature_acts.reshape(17,15, 21, 10)
# %%


result_matrix = torch.empty((17, 15))

for i in range(17):
    for j in range(15):
        # Extract the 2D tensor corresponding to the current i, j
        current_tensor = modified_feature_acts[i, j]
        
        # Apply the operation
        avg_gradient = torch.mean(torch.diff(torch.mean(current_tensor, dim=0)) / 0.1)
        
        # Store the result
        result_matrix[i, j] = avg_gradient


plt.imshow(result_matrix.detach().numpy())
plt.show()
# %%
plt.imshow(grad_matrix.cpu())
plt.show()
# %%
print(grad_matrix)
# %%
