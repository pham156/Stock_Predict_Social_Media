from statsmodels.tsa.arima.model import ARIMA
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        x = self.dropout(last_hidden)
        x = self.relu(self.fc1(x))
        pred = self.out(x).squeeze(-1)   # (batch,)
        return pred

class SeqDataset(Dataset):
    def __init__(self, X, y, task="cls"):
        self.X = torch.tensor(X, dtype=torch.float32)
        if task == "cls":
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)
        self.task = task

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def ARIMA_rolling_forecast(train, predict, dataset):
    p, d, q = 1, 0, 1

    history = list(train)
    preds_predict = []

    for t in range(len(predict)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        # get and append predict price y_hat
        yhat = model_fit.forecast(steps=1)[0]
        preds_predict.append(yhat)
        # add true price y to history to improve predict
        history.append(predict[t])

    mse = mean_squared_error(predict, preds_predict)
    print(f"Rolling MSE for ARIMA in {dataset} is: {mse}")

def run_ARIMA(dataset):
    series = dataset["return"].dropna().values

    n = len(series)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train, val, test = series[:train_end], series[train_end:val_end], series[val_end:]

    ARIMA_rolling_forecast(train, val, "val")
    train_plus_val = np.concatenate([train, val])
    ARIMA_rolling_forecast(train_plus_val, test, "test")

def build_sequences(X, y, dates, seq_len):
    X_seq, y_seq, date_seq = [], [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
        date_seq.append(dates[i+seq_len])
    return np.array(X_seq), np.array(y_seq), np.array(date_seq)


df = pd.read_csv("sp500_features.csv", parse_dates=["date"])

df_sym = df[df["ticker"] == "AAPL"].copy()
df_sym = df_sym.sort_values("date").reset_index(drop=True)

df_sym["return"] = df_sym["close"].pct_change()
df_sym["y_reg"] = df_sym["return"].shift(-1)

# increasing / decreasing/ same
eps = 0.0005

cond_up   = df_sym["y_reg"] > eps
cond_down = df_sym["y_reg"] < -eps

# 0 = decreasing, 1 = same, 2 = increasing
df_sym["y_3cls"] = np.select(
    [cond_down, cond_up],
    [0,         2],
    default=1
)

df_sym = df_sym.dropna().reset_index(drop=True)

feature_cols = [
    "open", "high", "low", "close", "volume",
    "sma_10", "ema_10", "rsi_14", "macd",
]

X = df_sym[feature_cols].values
y_reg = df_sym["y_reg"].values
y_3cls = df_sym["y_3cls"].values
dates = df_sym["date"].values

train_mask = (df_sym["date"] < "2022-01-01")
val_mask   = (df_sym["date"] >= "2022-01-01") & (df_sym["date"] < "2023-01-01")
test_mask  = (df_sym["date"] >= "2023-01-01")

X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
y_reg_train, y_reg_val, y_reg_test = y_reg[train_mask], y_reg[val_mask], y_reg[test_mask]
y_cls_train, y_cls_val, y_cls_test = y_3cls[train_mask], y_3cls[val_mask], y_3cls[test_mask]


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)
y_val_pred = lin_reg.predict(X_val)
y_test_pred = lin_reg.predict(X_test)

print("Linear Regression Val MSE:", mean_squared_error(y_reg_val, y_val_pred))
print("Linear Regression Test MSE:", mean_squared_error(y_reg_test, y_test_pred))

log_reg = LogisticRegression(
    max_iter=500,
    solver="lbfgs"
)
log_reg.fit(X_train, y_cls_train)

y_val_pred_cls = log_reg.predict(X_val)
y_test_pred_cls = log_reg.predict(X_test)

print("LogReg Val Acc:", accuracy_score(y_cls_val, y_val_pred_cls))

seq_len = 90
X_seq_cls, y_seq_cls, date_seq_cls = build_sequences(X, y_3cls, dates, seq_len)
X_seq_reg, y_seq_reg, date_seq_reg = build_sequences(X, y_reg, dates, seq_len)


train_mask = date_seq_cls < np.datetime64("2022-01-01")
val_mask   = (date_seq_cls >= np.datetime64("2022-01-01")) & (date_seq_cls < np.datetime64("2023-01-01"))
test_mask  = date_seq_cls >= np.datetime64("2023-01-01")


X_train_seq_cls, X_val_seq_cls, X_test_seq_cls = X_seq_cls[train_mask], X_seq_cls[val_mask], X_seq_cls[test_mask]
y_train_seq_cls, y_val_seq_cls, y_test_seq_cls = y_seq_cls[train_mask], y_seq_cls[val_mask], y_seq_cls[test_mask]

X_train_seq_reg, X_val_seq_reg, X_test_seq_reg = X_seq_reg[train_mask], X_seq_reg[val_mask], X_seq_reg[test_mask]
y_train_seq_reg, y_val_seq_reg, y_test_seq_reg = y_seq_reg[train_mask], y_seq_reg[val_mask], y_seq_reg[test_mask]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

# classification dataloader
train_ds_cls = SeqDataset(X_train_seq_cls, y_train_seq_cls, task="cls")
val_ds_cls   = SeqDataset(X_val_seq_cls,   y_val_seq_cls,   task="cls")
test_ds_cls  = SeqDataset(X_test_seq_cls,  y_test_seq_cls,  task="cls")

train_loader_cls = DataLoader(train_ds_cls, batch_size=batch_size, shuffle=True)
val_loader_cls   = DataLoader(val_ds_cls,   batch_size=batch_size, shuffle=False)
test_loader_cls  = DataLoader(test_ds_cls,  batch_size=batch_size, shuffle=False)

# regression dataloader
train_ds_reg = SeqDataset(X_train_seq_reg, y_train_seq_reg, task="reg")
val_ds_reg   = SeqDataset(X_val_seq_reg,   y_val_seq_reg,   task="reg")
test_ds_reg  = SeqDataset(X_test_seq_reg,  y_test_seq_reg,  task="reg")

train_loader_reg = DataLoader(train_ds_reg, batch_size=batch_size, shuffle=True)
val_loader_reg   = DataLoader(val_ds_reg,   batch_size=batch_size, shuffle=False)
test_loader_reg  = DataLoader(test_ds_reg,  batch_size=batch_size, shuffle=False)

n_features = X_train_seq_cls.shape[-1]

model_reg = LSTMRegressor(input_dim=n_features, hidden_dim=128, num_layers=5).to(device)

criterion_reg = nn.MSELoss()
optimizer_reg = optim.Adam(model_reg.parameters(), lr=1e-3)

epochs = 20

for epoch in range(epochs):
    model_reg.train()
    train_losses = []
    for X_batch, y_batch in train_loader_reg:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_reg.zero_grad()
        preds = model_reg(X_batch)          # (batch,)
        loss = criterion_reg(preds, y_batch)
        loss.backward()
        optimizer_reg.step()
        train_losses.append(loss.item())

    # 验证
    model_reg.eval()
    val_losses = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader_reg:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model_reg(X_batch)
            loss = criterion_reg(preds, y_batch)
            val_losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # calculate MSE
    val_mse = np.mean((np.array(all_preds) - np.array(all_labels))**2)

    print(f"[REG] Epoch {epoch+1}/{epochs} "
          f"TrainLoss={np.mean(train_losses):.6f} "
          f"ValMSE={val_mse:.6f}")


model_reg.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader_reg:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model_reg(X_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_mse = np.mean((np.array(all_preds) - np.array(all_labels))**2)
print("LSTM REG Test MSE:", test_mse)


# print("Train y_reg std:", y_reg_train.std())
# print("Val   y_reg std:", y_reg_val.std())
# print("Test  y_reg std:", y_reg_test.std())