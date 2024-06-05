# Each dataset contains train and validation folders
# The ZIP files roughly looks like this:
# -/
#   |- articles.parquet # Contains the articles
#   |- train/
#   |   |- behaviours.parquet # Contains the behaviours of the users
#   |   |- history.parquet # Contains the user information
#   |- validation/ # Looks the same as train

CHALLENGE_DATASET = {
    "demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    "large": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    "test": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
    "word2vec": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
    "image_embeddings": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip",
    "contrastive_vector": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip",
    "google-bert-base-multilingual-cased": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip",
    "FacebookAI-xlm-roberta-base": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip"
}