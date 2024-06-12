from recsys.model import MTRec, apply_softmax_crossentropy
from recsys.dataset import NewsDataset, load_data
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from ebrec.evaluation.metrics_protocols import MetricEvaluator
from ebrec.evaluation.metrics_protocols import AucScore, MrrScore, NdcgScore, LogLossScore, RootMeanSquaredError, AccuracyScore, F1Score
from recsys.metrics import calculate_accuracy, f1_score
import torch
import torch.nn.functional as F


def arg_list():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--bs", "--batch_size", type=int, default=32)
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("--wd", "--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", "--data", type=str, default="data")
    parser.add_argument("--embeddings_path", type=str, default="embeddings")
    # ../dataset/data/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet
    return parser.parse_args()


def main():
    args = arg_list()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu")

    train_dataset = load_data(None, args.data_path, "train", args.embeddings_path, batch_size=args.bs)
    val_dataset = load_data(None, args.data_path, "validation", args.embeddings_path, batch_size=args.bs)
    model = MTRec(args.hidden_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = 0
    for epoch in range(args.epochs):
        print(f"--- {epoch} / {args.epochs} ---")
        model.train()
        
        with tqdm(train_dataset) as t:
            for history, candidates, labels, repeats in t:
                history = history.to(device)
                candidates = candidates.to(device)
                labels = labels.to(device)
                scores = model(history, candidates)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), steps)
                writer.flush()
                t.set_postfix(loss=loss.item())
                steps = steps + 1
                
        model.eval()
        eval_scores = {"accuracy": 0, "f1": 0}
        with tqdm(val_dataset) as t:
            for history, candidates, labels, rep in t:
                with torch.no_grad():
                    history = history.to(device)
                    candidates = candidates.to(device)
                    labels = labels.to(device).flatten()
                    output = model(history, candidates).flatten()
                    acc = calculate_accuracy(output, rep, labels)
                    eval_scores["accuracy"] += acc.item()
                    #eval_scores["f1"] += f1_score(labels, output)
            eval_scores["accuracy"] /= len(val_dataset)
            #eval_scores["f1"] /= len(val_dataset)
            writer.add_scalar("Accuracy/val", eval_scores["accuracy"], steps)
            writer.add_scalar("F1/val", eval_scores["f1"], steps)
            writer.flush()
    writer.close()
    

# def main():
#     args = arg_list()
#     device = torch.device("cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu")

#     train_dataset = load_data(None, args.data_path, "train", args.embeddings_path, batch_size=args.bs)
#     val_dataset = load_data(None, args.data_path, "validation", args.embeddings_path, batch_size=args.bs)
#     model = MTRec(args.hidden_dim)
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.0, steps_per_epoch=len(train_dataset), epochs=args.epochs)
#     steps = 0
#     for epoch in range(args.epochs):
#         print(f"--- {epoch} / {args.epochs} ---")
#         model.train()
        
#         with tqdm(train_dataset) as t:
#             for history, candidates, labels, repeats in t:
#                 history = history.to(device)
#                 candidates = candidates.to(device)
#                 labels = labels.to(device).flatten()
#                 optimizer.zero_grad()
#                 output = model(history, candidates).flatten()
#                 loss = apply_softmax_crossentropy(output, repeats, labels)
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
#                 writer.add_scalar("Loss/train", loss.item(), steps)
#                 writer.flush()
#                 t.set_postfix(loss=loss.item())
#                 steps = steps + 1
                
#         model.eval()
#         eval_scores = {"accuracy": 0, "f1": 0}
#         with tqdm(val_dataset) as t:
#             for history, candidates, labels, rep in t:
#                 with torch.no_grad():
#                     history = history.to(device)
#                     candidates = candidates.to(device)
#                     labels = labels.to(device).flatten()
#                     output = model(history, candidates).flatten()
#                     acc = calculate_accuracy(output, rep, labels)
#                     eval_scores["accuracy"] += acc.item()
#                     #eval_scores["f1"] += f1_score(labels, output)
#             eval_scores["accuracy"] /= len(val_dataset)
#             #eval_scores["f1"] /= len(val_dataset)
#             writer.add_scalar("Accuracy/val", eval_scores["accuracy"], steps)
#             writer.add_scalar("F1/val", eval_scores["f1"], steps)
#             writer.flush()
#     writer.close()

if __name__ == "__main__":
    main()
