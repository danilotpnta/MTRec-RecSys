import argparse

from pytorch_lightning import Trainer, seed_everything

from recsys.dataset import NewsDataModule
from recsys.model import BERTMultitaskRecommender


def arg_list():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--bs", "--batch_size", type=int, default=512)
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-1)
    parser.add_argument("--wd", "--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--data_path",
        "--data",
        type=str,
        default="data",
        help="Path to the data directory containing the dataset and embeddings. NOTE: If the dataset is not present, it will be downloaded.",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="demo")
    parser.add_argument("--embeddings_type", type=str, default="xlm-roberta-base")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_gradient_surgery", action="store_true")
    return parser.parse_args()


def main():
    args = arg_list()
    print(args)

    # Set seed
    seed_everything(args.seed)

    datamodule = NewsDataModule(
        args.data_path,
        batch_size=args.bs,
        dataset=args.dataset,
        embeddings=args.embeddings_type,
        num_workers=args.num_workers,
        padding_value=None
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=1,
        # gradient_clip_val=0.3,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision="bf16-mixed",
        log_every_n_steps=1,
        profiler="advanced"
    )

    if args.load_from_checkpoint:
        model = BERTMultitaskRecommender.load_from_checkpoint(args.load_from_checkpoint)
    else:
        model = BERTMultitaskRecommender(lr=args.lr, wd=args.wd)
        datamodule.prepare_data()
        datamodule.setup()

        # model = MultitaskRecommender(
        #     args.hidden_dim,
        #     nhead=args.nhead,
        #     num_layers=args.num_layers,
        #     n_categories=datamodule.train_dataset.max_categories,
        #     lr=args.lr,
        #     wd=args.wd,
        #     use_gradient_surgery=args.use_gradient_surgery,
        # )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)

    # Make predictions on the test set
    # preds = trainer.test(model, datamodule=datamodule)
    # print(preds)


if __name__ == "__main__":
    main()
