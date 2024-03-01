import argparse
from pathlib import Path

import numpy as np
import polars as pl

import torch
import pytorch_lightning

from model import GeoDataset, GeoModel

num_workers = 7


def predict(config):
    pl.set_random_seed(56)
    pytorch_lightning.seed_everything(56, workers=True)

    with open("./hexses_target.lst", "r") as f:
        hexses_target = [x.strip() for x in f.readlines()]

    item2id = np.load("./item2id.npy", allow_pickle=True)
    item2id = item2id.item()
    items_target = np.load("./items_target.npy")
    xs = np.load("./xs.npy")
    ys = np.load("./ys.npy")

    transactions = pl.read_parquet(config.input_path)
    transactions = transactions.with_columns(
        std=pl.when(pl.col("count") >= 2).then(pl.col("std")).otherwise(0)
    )

    test_dataset = GeoDataset(transactions, item2id, xs, ys)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )
    joint_model = GeoModel(items_target)
    trainer = pytorch_lightning.Trainer(accelerator="gpu")
    prediction = trainer.predict(joint_model, test_dataloader, ckpt_path="./model.ckpt")
    matrix = np.concatenate([p.numpy() for p in prediction])

    customer_ids = transactions.unique("customer_id", maintain_order=True)["customer_id"]
    submit = pl.DataFrame(
        [customer_ids] + [pl.Series(loc, matrix[:, i]) for i, loc in enumerate(hexses_target)]
    )
    submit.sort("customer_id").write_parquet(config.output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hexses-target-path", "-ht", type=Path, required=True)
    parser.add_argument("--hexses-data-path", "-hd", type=Path, required=True)
    parser.add_argument("--input-path", "-i", type=Path, required=True)
    parser.add_argument("--output-path", "-o", type=Path, required=True)
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
