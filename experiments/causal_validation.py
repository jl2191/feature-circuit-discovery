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
    "/tmp/feature-circuit-discovery/datasets/ioi/ioi_test_100.json", "rb"
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

upstream_layer = 2
downstream_layer = 4
upstream_features = get_active_features_in_layer(tokenized_prompts, model, upstream_layer)
downstream_features = get_active_features_in_layer(tokenized_prompts, model, downstream_layer)

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
plt.imshow(grad_matrix.cpu().T)
plt.colorbar()
plt.show()
# %%
#Here we attempt to validate or prediction by seeing what kind of impact the ablation/reinforcement of certain feature activations may have.
upstream_sae = load_sae(canonical_sae_filenames[upstream_layer], device)
downstream_sae = load_sae(canonical_sae_filenames[downstream_layer], device)
scalars = [-10+i for i in range(21)]
#scalars = [0.0 for i in range(21)]
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
mat = []
for i in modified_feature_acts:
    stacked = torch.stack(i) #shape: scalars, batch, ds_features
    batch_mean = torch.mean(stacked, dim = 1).detach().cpu()
    differences = torch.diff(batch_mean, dim = 0)
    meaned_differences = torch.mean(differences, dim = 0)
    mat.append(meaned_differences)
plt.imshow(mat)
plt.colorbar()
plt.show()

# %%
stacked = torch.stack(modified_feature_acts[0])
print(stacked.shape)
p = torch.mean(stacked, dim = 1).detach().cpu()
print(p.shape)
diffs = torch.diff(p[:10])
#plt.plot(p[0])
plt.plot(p[10:])
plt.yscale("log")
plt.show()

# %%
#here i'm trying to see whether there is something wrong with the augmented forwardpass method by extracting the activations at which they are supposed to be augmented to see whether they change in the way i expect them to.
def validate_feature_modifcation_forwardpass(upstream_feature):
    """
    checks whether the addition of a feature vector to a residual stream actuall has the intended effect of strengthening the activation of the respective feature by comparing feature activations from before and after the addition of the feature vector.
    """
    _ , unaugmented_acts = model_with_added_feature_vector(model, tokenized_prompts, upstream_sae, upstream_layer, upstream_feature, 0, upstream_layer)
    unaugmented_feature_acts = upstream_sae.encode(unaugmented_acts)[:, -1, upstream_features]
    _ , augmented_acts = model_with_added_feature_vector(model, tokenized_prompts, upstream_sae, upstream_layer, upstream_feature, 1, upstream_layer)
    augmented_feature_acts = upstream_sae.encode(augmented_acts)[:, -1, upstream_features]

    diffs = unaugmented_feature_acts - augmented_feature_acts
    print(diffs.shape)
    plt.imshow(diffs.detach().cpu())
    plt.show()
for i in upstream_features:
    validate_feature_modifcation_forwardpass(i)
# %%$
#checking whether zero a scalar actually produced the same result as the neutral method.

neutral_logits = model(tokenized_prompts).logits.detach()
zero_scalar_logits, _ = model_with_added_feature_vector(model, tokenized_prompts, upstream_sae, upstream_layer, upstream_features[0], 0, upstream_layer)
print("the difference between the two logits should be zero.\nLogit difference:",float(torch.sum(torch.abs(neutral_logits-zero_scalar_logits))))
# %%
