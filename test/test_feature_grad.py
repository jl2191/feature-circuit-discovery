# import json

# with open("ioi_test_100.json", "rb") as file:
#     prompt_data = json.load(file)

# inputs = tokenizer(
#     prompt_data, return_tensors="pt", add_special_tokens=True, padding=True
# ).data["input_ids"].to(device)
# matrices = []
# for layer in tqdm(range(len(activated_features)-1)):
#   grad_matrix = compute_gradient_matrix(inputs, layer, layer+1, activated_features[layer], activated_features[layer+1], verbose = True)
#   matrices.append(grad_matrix)
