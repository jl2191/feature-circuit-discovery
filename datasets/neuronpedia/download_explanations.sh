#!/bin/bash

# Define the base URL for the neuronpedia S3 directory
BASE_URL="https://neuronpedia-exports.s3.amazonaws.com"
MODEL_ID="gemma-2-2b" # the model series you want to download
SAE_ID="gemmascope-res-16k" # the sae ids you want to download
# this will download the sae explanations for all the layers

# Download the XML file listing all files in the directory
file_list=$(wget -q -O - "${BASE_URL}/?prefix=explanations-only/")

# Parse the XML to extract file names
files=$(echo "$file_list" | grep -oP "(?<=<Key>explanations-only/)[^<]+")

# Create a directory to store the downloaded files
mkdir -p "explanations"

# Loop through each file name and download it if it matches the model_id and sae_id
for file in $files; do
    if [[ $file == ${MODEL_ID}_*${SAE_ID}.json ]]; then
        wget -q "${BASE_URL}/explanations-only/${file}" -P "explanations/"
    fi
done

echo "Download complete. Files are saved in the 'explanations' directory."