import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

all_data = pd.read_csv("analysis_data.csv", index_col=0, encoding="shift-jis")

# データをPyTorchでの学習に利用できる形式に変換
# "tip"の列を目的にする
target = torch.tensor(all_data["若年層人口"].values.reshape(-1, 1), dtype=torch.float32)
# "tip"以外の列を入力にする
input = torch.tensor(
    all_data.drop(["year", "area", "code", "若年層人口"], axis=1).values.astype(np.float32), dtype=torch.float32
)

# データセットの作成
all_data_dataset = torch.utils.data.TensorDataset(input, target)

# 学習データ、検証データ、テストデータに 6:2:2 の割合で分割
train_size = int(0.6 * len(all_data_dataset))
val_size = int(0.2 * len(all_data_dataset))
test_size = len(all_data_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    all_data_dataset, [train_size, val_size, test_size]
)

# バッチサイズ：25として学習用データローダを作成
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)
# 検証用ローダ作成
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25)
# テスト用ローダを作成
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25)


class Net(L.LightningModule):
    # New: バッチサイズ等を引数に指定
    def __init__(self, input_size=35, hidden_size=30, output_size=1, batch_size=25):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    # 変更なし
    def forward(self, x):
        h = torch.tanh(self.l1(x))
        o = self.l2(h)
        return o

    # New: 目的関数の設定
    def lossfun(self, y, t):
        return F.mse_loss(y, t)

    # New: optimizer の設定
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)

    # New: 学習データに対する処理
    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        self.log("val_loss", loss, prog_bar=True)
        return loss


model = Net()
logger = CSVLogger("lightning_logs", name="my_model")
trainer = Trainer(max_epochs=100, accelerator="mps", logger=logger)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
