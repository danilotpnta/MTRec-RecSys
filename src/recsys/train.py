import argparse
import pickle

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from recsys.dataset import NewsDataModule
from recsys.model import BERTMultitaskRecommender, MultitaskRecommender
from ebrec.utils._python import write_submission_file


def arg_list():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--bs", "--batch_size", type=int, default=512)
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-1)
    parser.add_argument("--wd", "--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--data_path",
        "--data",
        type=str,
        default="data",
        help="Path to the data directory containing the dataset and embeddings. NOTE: If the dataset is not present, it will be downloaded.",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="demo")
    parser.add_argument("--embeddings_type", type=str, default="xlm-roberta-base")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_gradient_surgery", action="store_true")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--use_precomputed_embeddings", action="store_true")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=("bf16", "bf16-mixed", "32", "16", "16-mixed"),
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--disable_category", action="store_true")
    parser.add_argument("--disable_sentiment", action="store_true")
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
        max_length=args.max_length,
        padding_value=0,
        dataset_type="v1" if args.use_precomputed_embeddings else "v2",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    profiler = AdvancedProfiler(filename="profiler_results.txt")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.dataset}-bs{args.bs}-{args.use_lora and 'use_lora'}-{args.seed}-lr{args.lr}",
        filename="{epoch:02d}-{step:02d}",
        save_top_k=-1,
        every_n_train_steps=2000,
        save_weights_only=False,
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=1,
        # gradient_clip_val=0.3,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=args.precision,
        log_every_n_steps=1,
        # profiler=profiler,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=TensorBoardLogger("lightning_logs", name="bert_recommender"),
        # strategy="ddp_find_unused_parameters_true",
    )

    datamodule.prepare_data()
    datamodule.setup()

    if not args.use_precomputed_embeddings:
        if args.load_from_checkpoint:
            model = BERTMultitaskRecommender.load_from_checkpoint(
                args.load_from_checkpoint
            )
        else:
            model = BERTMultitaskRecommender(
                epochs=args.epochs,
                lr=args.lr,
                wd=args.wd,
                batch_size=args.bs,
                steps_per_epoch=datamodule.train_dataset.__len__() // args.bs,
                use_gradient_surgery=args.use_gradient_surgery,
                n_categories=datamodule.train_dataset.max_categories,
                sentiment_labels=datamodule.train_dataset.max_sentiment_labels,
                use_lora=args.use_lora,
                disable_category=args.disable_category,
                disable_sentiment=args.disable_sentiment,
            )
    else:
        if args.load_from_checkpoint:
            model = MultitaskRecommender.load_from_checkpoint(args.load_from_checkpoint)
        else:
            embeddings = datamodule.train_dataset.lookup_matrix.detach().clone()
            model = MultitaskRecommender(
                args.hidden_dim,
                nhead=args.nhead,
                num_layers=args.num_layers,
                n_categories=datamodule.train_dataset.max_categories,
                lr=args.lr,
                wd=args.wd,
                use_gradient_surgery=args.use_gradient_surgery,
                batch_size=args.bs,
                embeddings=embeddings,
                steps_per_epoch=datamodule.train_dataset.__len__() // args.bs,
            )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
