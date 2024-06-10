import argparse
import os
import zipfile

import requests
from tqdm import tqdm

CHALLENGE_DATASET = {
    "demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    "large": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    "test": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
    "word2vec": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
    "image_embeddings": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip",
    "contrastive_vector": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip",
    "google-bert-base-multilingual-cased": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip",
    "FacebookAI-xlm-roberta-base": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip",
}


def download_file(url: str, dest: str):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 Mebibyte

    file = open(dest, "wb")
    for data in tqdm(
        response.iter_content(block_size),
        total=total_size // block_size,
        unit="MiB",
        unit_scale=True,
        unit_divisor=1024,
    ):
        file.write(data)
    file.close()

    return dest


def unzip_file(src: str, dest: str):
    with zipfile.ZipFile(src) as zip_ref:
        zip_ref.extractall(dest)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="demo", choices=CHALLENGE_DATASET.keys()
    )
    parser.add_argument("--save_dir", type=str, default="dataset/data")
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.dataset)
    url = CHALLENGE_DATASET[args.dataset]
    filename = url.split("/")[-1]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Download and unzip the dataset
    zipfile = download_file(url, os.path.join(save_dir, filename))

    path = unzip_file(zipfile, save_dir)

    print(f"Dataset downloaded and saved to {path}")


if __name__ == "__main__":
    main()
