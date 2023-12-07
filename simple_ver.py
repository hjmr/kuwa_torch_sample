import numpy as np
import pandas as pd

import torch

all_data = pd.read_csv("analysis_data.csv", index_col=0, encoding="shift-jis")

device_name = "mps"
if not torch.backends.mps.is_available():
    device_name = "cpu"
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not " "built with MPS enabled.")
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
device_name = "cuda:0" if torch.cuda.is_available() else device_name
print(f"Using device: {device_name}")
device = torch.device(device_name)

# データをPyTorchでの学習に利用できる形式に変換
# "tip"の列を目的にする
target = torch.tensor(all_data["若年層人口"].values.reshape(-1, 1), dtype=torch.float32, device=device)
# "tip"以外の列を入力にする
input = torch.tensor(
    all_data.drop(["year", "area", "code", "若年層人口"], axis=1).values.astype(np.float32),
    dtype=torch.float32,
    device=device,
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


# 3層順方向ニューラルネットワークモデル
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        o = self.l2(h)
        return o


# NNのオブジェクトを作成
model = SimpleNN(35, 30, 1).to(device)
# オプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

# 損失の可視化のために各エポックの損失を保持しておくリスト
train_loss_list = []
test_loss_list = []
# データセット全体に対して10000回学習
for epoch in range(1000):
    epoch_loss = []
    # バッチごとに学習する
    for x, y_hat in train_loader:
        y = model(x)
        train_loss = torch.nn.functional.mse_loss(y, y_hat)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        epoch_loss.append(train_loss)
    train_loss_list.append(torch.tensor(epoch_loss).mean())

    with torch.inference_mode():  # 推論モード（学習しない）
        y = model(input)
        test_loss = torch.nn.functional.mse_loss(y, target)
        test_loss_list.append(test_loss.mean())

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, train_loss: {train_loss_list[-1]}, test_loss: {test_loss_list[-1]}")
