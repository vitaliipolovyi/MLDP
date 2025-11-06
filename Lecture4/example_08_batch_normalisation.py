# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Дані ---
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
X_train_scaled_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader_raw = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
train_loader_scaled = DataLoader(TensorDataset(X_train_scaled_t, y_train_t), batch_size=32, shuffle=True)

# %%
class MLP(nn.Module):
    def __init__(self, input_dim, use_bn=False):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 2)
        # )
        if use_bn:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
    def forward(self, x):
        return self.model(x)

# %%
def train_model(model, train_loader, X_test_t, y_test_t, epochs=200, lr=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    acc_history = []
    X_test_t, y_test_t = X_test_t.to(device), y_test_t.to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        # оцінка
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_test_t), dim=1)
            acc = (preds == y_test_t).float().mean().item()
            acc_history.append(acc)
    return acc_history

# %%
model_raw = MLP(X_train.shape[1], use_bn=False)
model_scaled = MLP(X_train.shape[1], use_bn=False)
model_bn = MLP(X_train.shape[1], use_bn=True)

# %%
acc_raw = train_model(model_raw, train_loader_raw, X_test_t, y_test_t)
acc_scaled = train_model(model_scaled, train_loader_scaled, X_test_scaled_t, y_test_t)
acc_bn = train_model(model_bn, train_loader_scaled, X_test_scaled_t, y_test_t)

# %%
plt.figure(figsize=(8,5))
plt.plot(acc_raw, label='Без нормалізації')
plt.plot(acc_scaled, label='StandardScaler')
plt.plot(acc_bn, label='BatchNorm')
plt.xlabel('Епоха')
plt.ylabel('Accuracy на тесті')
plt.title('Порівняння нормалізацій на breast_cancer dataset')
plt.legend()
plt.grid(True)
plt.show()
