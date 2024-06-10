from recsys.model import MTRec
from recsys.dataset import NewsDataset, load_data
import argparse
from tqdm import tqdm

import torch


def arg_list():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--bs", "--batch_size", type=int, default=32)
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("--wd", "--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", "--data", type=str, default="data")
    parser.add_argument("--embeddings_path", type=str, default="embeddings")
    # ../dataset/data/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet
    return parser.parse_args()


def main():
    args = arg_list()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu")

    train_dataset = load_data(None, args.data_path, "train", args.embeddings_path)
    model = MTRec(args.hidden_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        print(f"--- {epoch} / {args.epochs} ---")
        model.train()
        with tqdm(train_dataset) as t:
            for history, candidates, labels in t:
                history = history.to(device)
                candidates = candidates.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = model(history, candidates)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=loss.item())


if __name__ == "__main__":
    main()
