#!/bin/bash

# check if 'ckp' directory exists, create it if not
if [ ! -d "ckp" ]; then
    mkdir ckp
fi

# download encoder.pkl and decoder.pkl files using wget
url1="https://cdn.openai.com/dall-e/encoder.pkl"
url2="https://cdn.openai.com/dall-e/decoder.pkl"
echo "Downloading files from $url1 and $url2..."

# run wget commands in parallel using 2 threads
wget -P ckp -q --show-progress --tries=10 --retry-connrefused $url1 &
wget -P ckp -q --show-progress --tries=10 --retry-connrefused $url2 &

# wait for the downloads to finish
wait

echo "Downloads complete."
