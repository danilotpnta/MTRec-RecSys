import requests
from tqdm import tqdm
import os

# List of files to download
files = [
    ("ebnerd_demo.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip"),
    ("ebnerd_small.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip"),
    ("ebnerd_large.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip"),
    ("articles_large_only.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip"),
    ("ebnerd_testset.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"),
    ("predictions_large_random.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/predictions_large_random.zip"),
    ("Ekstra_Bladet_word2vec.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip"),
    ("Ekstra_Bladet_image_embeddings.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip"),
    ("Ekstra_Bladet_contrastive_vector.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip"),
    ("google_bert_base_multilingual_cased.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip"),
    ("FacebookAI_xlm_roberta_base.zip", "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip"),
]

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    with open(dest, 'wb') as file, tqdm(
            desc=dest,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

# Create a directory for the downloads if it doesn't exist
os.makedirs("downloads", exist_ok=True)

# Download all files
for filename, url in files:
    destination = os.path.join("downloads", filename)
    download_file(url, destination)

print("All files have been downloaded.")
