import polars as pl

import torch
from torch import nn
import pytorch_lightning

from transformers import ResNetConfig, ResNetModel

map_size = 128


class GeoDataset(torch.utils.data.Dataset):
    def __init__(self, transactions, item2id, xs, ys):
        super().__init__()

        transactions = transactions.with_columns(pl.col("h3_09").map_dict(item2id, default=None))

        self._data = (
            transactions
            .group_by("customer_id", maintain_order=True)
            .agg(
                items=pl.col("h3_09"),
                counts=pl.col("count"),
            )
        )
        self._xs = xs
        self._ys = ys

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        row = self._data.row(index)
        items = row[1]
        counts = row[2]

        pixel_values = torch.zeros((map_size, map_size, 3)).float()
        for i, item in enumerate(items):
            pixel_values[self._xs[item]][self._ys[item]][0] = 1.0
            pixel_values[self._xs[item]][self._ys[item]][1] += 1.0
            pixel_values[self._xs[item]][self._ys[item]][2] += counts[i]
        pixel_values[:, :, 2] = torch.log1p(pixel_values[:, :, 2])

        return pixel_values


class GeoModel(pytorch_lightning.LightningModule):
    def __init__(self, items_target):
        super().__init__()

        self.preprocess = nn.Sequential(
            nn.Linear(3, 12, False),
            nn.ReLU(),
            nn.Linear(12, 12, False),
            nn.ReLU(),
            nn.Linear(12, 3, False),
        )
        config = ResNetConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[128, 256, 512, 1024],
            depths=[2, 2, 2, 2],
            layer_type='bottleneck',
            hidden_act='relu',
        )
        self.resnet = ResNetModel(config)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], len(items_target)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.preprocess(x)
        output = self.resnet(x.permute(0, 3, 1, 2))
        return self.classifier(output.pooler_output)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch
        prediction_scores = self.forward(pixel_values)
        return prediction_scores.detach().cpu()
