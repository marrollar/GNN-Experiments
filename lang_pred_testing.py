import os

os.add_dll_directory("D:\\Programming\\External Libraries\\CUDA\\11.8\\bin")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.loader import NodeLoader, DataLoader, GraphSAINTSampler, GraphSAINTNodeSampler

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Read input data '''

edges_df = pd.read_csv("data/large_twitch_edges.csv")
features_df = pd.read_csv("data/large_twitch_features.csv")

features_df["created_at"] = pd.to_datetime(features_df["created_at"])
features_df["updated_at"] = pd.to_datetime(features_df["updated_at"])

''' Filter accounts which haven't been active recently '''

q1, q2, q3 = np.quantile(features_df["updated_at"], q=[0.25, 0.5, 0.75])
iqr = q3 - q1
lower, upper = q2 - iqr, q2 + iqr

# filtered_df = features_df[(features_df["updated_at"] > lower) & (features_df["updated_at"] < upper)]
#
# remaining_ids = filtered_df["numeric_id"]
# filtered_edges_df = edges_df[
#     edges_df["numeric_id_1"].isin(remaining_ids.values) & edges_df["numeric_id_2"].isin(remaining_ids.values)]

filtered_df = features_df
filtered_edges_df = edges_df

''' Normalize data '''


def normalize(series):
    return (series - series.mean()) / series.std()


mms_date = MinMaxScaler()
mms_life = MinMaxScaler()

normed_df = filtered_df.drop(columns=["numeric_id", "updated_at"])

# tmp = pd.concat([normed_df["created_at"], normed_df["updated_at"]])
# mms_date.fit(tmp.to_numpy().reshape(-1, 1))

normed_df["views"] = normalize(normed_df["views"])
normed_df["life_time"] = normalize(normed_df["life_time"])
normed_df["created_at"] = normalize(normed_df["created_at"])
# normed_df["updated_at"] = normalize(normed_df["updated_at"])
# normed_df["life_time"] = mms_life.fit_transform(normed_df["life_time"].to_numpy().reshape(-1, 1))
# normed_df["created_at"] = mms_date.transform(normed_df["created_at"].to_numpy().reshape(-1, 1))
# normed_df["updated_at"] = mms_date.transform(normed_df["updated_at"].to_numpy().reshape(-1, 1))
# normed_df = pd.get_dummies(normed_df, columns=["language"], drop_first=True, dtype=int)

langs = normed_df["language"].unique()
lang_encoding = {k: v for k, v in zip(langs, range(len(langs)))}
normed_df["language"] = normed_df["language"].replace(lang_encoding)

normed_df.head()

''' Convert to tensors '''


def split_feats_classes(df, class_col_name):
    x = df.drop(columns=[class_col_name])
    y = df[class_col_name]

    return x, y


# normed_np = torch.tensor(normed_df.drop(columns=["mature"]).to_numpy()).float()
# edges_np = torch.tensor(filtered_edges_df.to_numpy())

# mature_ohe = pd.DataFrame({"mature": normed_df["mature"], "not_mature": 1-normed_df["mature"]})
# target_np = torch.tensor(mature_ohe.to_numpy())
# target_np = torch.tensor(normed_df["mature"].to_numpy())

x, y = split_feats_classes(normed_df, "language")
x_np = torch.tensor(x.to_numpy()).float()
y_np = torch.tensor(y.to_numpy())
edges_np = torch.tensor(filtered_edges_df.to_numpy())

''' Train Test Val split '''

node_idx = normed_df.index.values

train_idx, test_idx = train_test_split(node_idx, test_size=0.25)
val_idx, test_idx = train_test_split(test_idx, test_size=0.25)

train_mask = torch.tensor(normed_df.index.isin(train_idx))
val_mask = torch.tensor(normed_df.index.isin(val_idx))
test_mask = torch.tensor(normed_df.index.isin(test_idx))

''' Construct TorchGeo graph '''

graph = Data(x=x_np,
             y=y_np,
             edge_index=edges_np.t().contiguous(),
             train_mask=train_mask,
             val_mask=val_mask,
             test_mask=test_mask)
graph

''' Declare GCN architecture '''


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(graph.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        x = self.lin(x)

        return x


''' Training loop '''


def train(model, loader):
    model.train()

    total_loss = 0
    total_examples = 0

    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index.to(device))
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes
        total_examples += batch.num_nodes

    return total_loss / total_examples


def test(model, graph):
    model.eval()

    out = model(graph.x.to(device), graph.edge_index.to(device))
    preds = F.log_softmax(out, dim=-1).argmax(1)
    # print(f"Preds: {preds}, torch.unique(preds)")
    # print(f"True: {graph.y}")
    correct = preds.eq(graph.y.to(device))

    accs = []
    for _, mask in graph("train_mask", "val_mask", "test_mask"):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    return accs


''' Setup torch inputs and variables for training loop '''

num_classes = len(lang_encoding)
hc = 32

# model = GCN(hc, num_classes).to(device)
model = torch.load("best_model.pk").to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, momentum=0.9)

# loader = GraphSAINTNodeSampler(graph, batch_size=512, num_steps=16)
loader = [graph]

best_val_acc = 0
best_test_acc = 0

loss_hist = []
val_hist = []
test_hist = []

for epoch in range(10000):
    loss = train(model, loader)
    train_acc, val_acc, test_acc = test(model, graph)

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

        torch.save(model, f"best_model.pk")

    if epoch % 5 == 0:
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
        )

model = torch.load("best_model.pk")
model.eval()

out = model(graph.x.to(device), graph.edge_index.to(device))
preds = F.log_softmax(out, dim=-1).argmax(1)

np.savetxt("out.txt", out.detach().cpu().numpy(), fmt="%f")
np.savetxt("preds.txt", preds.cpu().numpy(), fmt="%d")
np.savetxt("y.txt", graph.y.cpu().to(int), fmt="%d")
