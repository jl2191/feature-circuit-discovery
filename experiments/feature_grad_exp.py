from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import torch
import gc
from tqdm import tqdm
import torch.nn as nn


model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map='auto',
)
device = model.device
tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b")

canonical_sae_filenames = [
    "layer_0/width_16k/average_l0_105/params.npz",
    "layer_1/width_16k/average_l0_102/params.npz",
    "layer_2/width_16k/average_l0_141/params.npz",
    "layer_3/width_16k/average_l0_59/params.npz",
    "layer_4/width_16k/average_l0_124/params.npz",
    "layer_5/width_16k/average_l0_68/params.npz",
    "layer_6/width_16k/average_l0_70/params.npz",
    "layer_7/width_16k/average_l0_69/params.npz",
    "layer_8/width_16k/average_l0_71/params.npz",
    "layer_9/width_16k/average_l0_73/params.npz",
    "layer_10/width_16k/average_l0_77/params.npz",
    "layer_11/width_16k/average_l0_80/params.npz",
    "layer_12/width_16k/average_l0_82/params.npz",
    "layer_13/width_16k/average_l0_84/params.npz",
    "layer_14/width_16k/average_l0_84/params.npz",
    "layer_15/width_16k/average_l0_78/params.npz",
    "layer_16/width_16k/average_l0_78/params.npz",
    "layer_17/width_16k/average_l0_77/params.npz",
    "layer_18/width_16k/average_l0_74/params.npz",
    "layer_19/width_16k/average_l0_73/params.npz",
    "layer_20/width_16k/average_l0_71/params.npz",
    "layer_21/width_16k/average_l0_70/params.npz",
    "layer_22/width_16k/average_l0_72/params.npz",
    "layer_23/width_16k/average_l0_75/params.npz",
    "layer_24/width_16k/average_l0_73/params.npz",
    "layer_25/width_16k/average_l0_116/params.npz"
]

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold).float()
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

def load_sae(filename):
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename=filename,
        force_download=False,
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    return sae.to(model.device)


def gather_residual_activations(model, target_layer, inputs):
    activations = {}
    def hook_fn(module, input, output):
        activations['output'] = output[0]

    handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
    model(inputs)
    handle.remove()

    return activations['output']


def find_frequent_nonzero_indices(tensor_list, threshold = 0.9):

    result_indices = []
    
    for tensor in tensor_list:
        non_zero_count = torch.sum(tensor != 0, dim=0)/tensor.shape[0]
        valid_indices = torch.where(non_zero_count >= threshold)[0]# Find the indices where the count of non-zero entries meets the threshold
        result_indices.append(valid_indices)# Append these indices to the result list
    return result_indices


def get_active_features(prompts, threshold = 0.9):
    
    #change this to tokenizer().[whatever attrivutes of the return object we need]
    tokenized_prompts = nputs = tokenizer(
        prompts, return_tensors="pt", add_special_tokens=True, padding=True
    ).data["input_ids"].to(device)
    feature_acts = []

    for layer in tqdm(range(model.config.num_hidden_layers)):
        sae = load_sae(canonical_sae_filenames[layer])
        activations = gather_residual_activations(model, layer, tokenized_prompts)
        sae_activations = sae.encode(activations)
        feature_acts.append(sae_activations)

    #channge this to finding the features that activate in most/all of the forward passes in rather than looking at all of the activated ones

    #find all the activated features in all the layers
    activated_features = find_frequent_nonzero_indices(feature_acts, threshold = 0.9)


#INSERT FEATURE DESCRIPTIONS HERE

def describe_feature(layer, idx):
    """
    Returns a string description of a a feature of a certain layer and index of gemma 2 2b
    """
    pass


def describe_features(layers, indices):
    """
    Describes multiple features of multiplae layers. layers and indices should therefore be 1 and 2 dimensional lists of integers.
    """
    pass




def compute_gradient_matrix(inputs, upstream_layer_idx, downstream_layer_idx, upstream_features, downstream_features, verbose = False):
    device = next(model.parameters()).device

    # Clear GPU cache and enable gradient computation
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_grad_enabled(True)

    if verbose:
      print()
      print(f"loading upstream sae (Layer {upstream_layer_idx})")
    sae_upstream = load_sae(canonical_sae_filenames[upstream_layer_idx])

    if verbose:
      print("upsteam sae loaded")
      print(f"loading downstram sae (Layer {downstream_layer_idx})")
    sae_downstream = load_sae(canonical_sae_filenames[downstream_layer_idx])  

    if verbose:
      print("downsteam sae loaded")
    m = len(upstream_features)
    n = len(downstream_features)

    d_model = sae_upstream.W_dec.size(1)

    # Define the scalar parameters 'a' as a tensor with requires_grad=True
    a = torch.zeros(m, requires_grad=True, device=device)  # Shape: (m,)

    feature_vectors = []
    if verbose:
      print(f"getting upstram feature vectors")
    for k in upstream_features:
        feature_vector = sae_upstream.W_dec[k, :]  
        feature_vectors.append(feature_vector)

    feature_vectors = torch.stack(feature_vectors)  

    def modify_residual_activations(module, input, output):
      residual = output[0] if isinstance(output, tuple) else output
      residual_modified = residual.clone()

      weighted_feature_vectors = (a.view(m, 1, 1) * feature_vectors.view(m, 1, d_model)).sum(dim=0)
      expanded_weighted_feature_vector = weighted_feature_vectors.expand(residual_modified.size(0), -1, -1)

      residual_modified[:, 0:1, :] += expanded_weighted_feature_vector

      return (residual_modified,) + output[1:] if isinstance(output, tuple) else residual_modified

    if verbose:
      print(f"conducting forward pass")
    hook = model.model.layers[upstream_layer_idx].register_forward_hook(modify_residual_activations)
    outputs = model(inputs, output_hidden_states=True)

    hook.remove()

    hidden_states = outputs.hidden_states
    act_upstream = hidden_states[upstream_layer_idx]
    act_downstream = hidden_states[downstream_layer_idx]


    sae_downstream_acts = sae_downstream.encode(act_downstream)

    batch_idx = 0
    seq_idx = 0

    features_downstream = sae_downstream_acts[batch_idx, seq_idx, downstream_features]

    gradient_matrix = torch.zeros(n, m, device=device)

    # Compute gradients for each feature in the downstream layer with respect to 'a'
    if verbose:
      print(f"computing gradients")
      for i in tqdm(range(n)):
          if a.grad is not None:
              a.grad.zero_()
          features_downstream[i].backward(retain_graph=True)
          gradient_matrix[i, :] = a.grad.detach()
          a.grad.zero_()
    else:
      for i in range(n):
          if a.grad is not None:
              a.grad.zero_()
          features_downstream[i].backward(retain_graph=True)
          gradient_matrix[i, :] = a.grad.detach()
          a.grad.zero_()
    if verbose:
      print(f"clean up")
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
      print(f"done")
    return gradient_matrix