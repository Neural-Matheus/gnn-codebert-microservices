# -*- coding: utf-8 -*-
## Instalação das lib's

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import pickle
with open("/content/call_graph_with_label.pkl", "rb") as f:
    call_graph = pickle.load(f)
print(f"✅ Loaded graph with {call_graph.number_of_nodes()} nodes and {call_graph.number_of_edges()} edges")

import networkx as nx
import statistics

density = nx.density(call_graph)
print(f"📈 Density: {density:.4f}")

isolates = list(nx.isolates(call_graph))
print(f"🗑️ Isolated nodes: {len(isolates)}")

undirected = call_graph.to_undirected()
n_components = nx.number_connected_components(undirected)
components = sorted(nx.connected_components(undirected), key=len, reverse=True)
largest = len(components[0]) if components else 0
print(f"🔗 Connected components: {n_components}")
print(f"   – Largest component size: {largest}")

degrees = [d for _, d in call_graph.degree()]
print(f"📐 Min degree: {min(degrees)}")
print(f"📐 Max degree: {max(degrees)}")
print(f"📐 Avg degree: {statistics.mean(degrees):.2f}")

top5 = sorted(call_graph.degree(), key=lambda x: x[1], reverse=True)[:5]
print("⭐ Top 5 nodes by degree:")
for node, deg in top5:
    print(f"   • {node}: {deg}")

first_node = next(iter(call_graph.nodes()))
print(f"\nFirst node: {first_node}")
print("Code snippet:\n")
print(call_graph.nodes[first_node]["code"])

import textwrap
import re

for src, dst in call_graph.edges():
    snippet = call_graph.nodes[src]["code"]
    svc_src, fn_src = src.split(":", 1)
    svc_dst, fn_dst = dst.split(":", 1)

    print(f"\n📌 Edge: {src} → {dst}")
    print(f"--- Source Function `{fn_src}` in service `{svc_src}` ---")
    code = textwrap.dedent(snippet)
    print(code)

    pattern = rf"\b{re.escape(fn_dst)}\s*\("
    matches = list(re.finditer(pattern, code))
    if matches:
        for m in matches:
            line_no = code[:m.start()].count("\n") + 1
            line = code.splitlines()[line_no-1]
            print(f"    -> calls `{fn_dst}` at line {line_no}: {line.strip()}")
    else:
        print(f"    ⚠️ No explicit call to `{fn_dst}` found in this snippet")

"""## Importação do GraphCodeBERT"""

from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

model.eval()

import torch

node_list   = list(call_graph.nodes())
node_labels    = [call_graph.nodes[n]["label"] for n in node_list]
node_snippets  = [call_graph.nodes[n]["code"]  for n in node_list]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = torch.tensor(node_labels, dtype=torch.long).to(device)

num_classes = int(labels.max()) + 1

print(num_classes)

cls_embeddings = []

from torch.utils.data import DataLoader

class SnippetDataset(torch.utils.data.Dataset):
    def __init__(self, snippets, tokenizer, max_length=512):
        self.snippets = snippets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        return self.snippets[idx]

    def collate_fn(self, batch):
        return self.tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

dataset = SnippetDataset(node_snippets, tokenizer)
loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn, num_workers=2)

cls_embs = []
model.to("cuda")
for batch in loader:
    batch = {k: v.to("cuda") for k,v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    hidden = outputs.last_hidden_state
    cls_embs.append(hidden[:, 0, :])
    mean_embs.append(hidden.mean(dim=1))
    max_embs.append(hidden.max(dim=1).values)

cls_embeddings = torch.cat(cls_embs, dim=0)

print(cls_embeddings)

import torch

node_list = list(call_graph.nodes())
name2idx  = {name: i for i, name in enumerate(node_list)}
edge_idx  = [
    [ name2idx[src], name2idx[dst] ]
    for src, dst in call_graph.edges()
]
edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
print("edge_index shape:", edge_index.shape)

print("Arestas no call_graph:", call_graph.number_of_edges())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch_geometric.data import Data

data = Data(
    x=node_features_cls,
    edge_index=edge_index,
    y=labels
)
print(data)

data = data.to(device)

torch.manual_seed(42)

from sklearn.model_selection import train_test_split

y_cpu = data.y.cpu().numpy()

idx = list(range(data.num_nodes))
train_idx, test_idx = train_test_split(
    idx,
    test_size=0.2,
    random_state=42,
    stratify=y_cpu
)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.train_mask[train_idx] = True
data.test_mask[test_idx]   = True

train_labels   = data.y[data.train_mask]
class_counts   = torch.bincount(train_labels)
class_weights  = 1.0 / class_counts.float()
class_weights /= class_weights.sum()
class_weights *= class_counts.numel()
class_weights = class_weights.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

class GATv2Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out   = model(data.x, data.edge_index)
    loss  = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs  = F.softmax(logits, dim=1)
    preds  = probs.argmax(dim=1)

    y_true = data.y.cpu().numpy()
    y_pred = preds.cpu().numpy()
    mask_train = data.train_mask.cpu().numpy()
    mask_test  = data.test_mask.cpu().numpy()

    metrics = {}
    for split, mask in [("train", mask_train), ("test", mask_test)]:
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1  = f1_score(y_true[mask], y_pred[mask], average="macro")
        y_oh = torch.nn.functional.one_hot(data.y, num_classes).cpu().numpy()
        auc  = roc_auc_score(y_oh[mask], probs.cpu().numpy()[mask], multi_class="ovr")
        metrics[split] = {"acc": acc, "f1": f1, "auc": auc}
    return metrics

results = {}
for ModelClass in (GCNNet, GATNet, GATv2Net):
    model     = ModelClass(in_channels=data.num_features, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(1, 101):
        loss = train_epoch(model, data, optimizer, criterion)
        if epoch % 20 == 0:
            print(f"{ModelClass.__name__} Epoch {epoch:03d} Loss {loss:.4f}")

    metrics = evaluate(model, data)
    results[ModelClass.__name__] = metrics
    print(f"\n=== {ModelClass.__name__} Results ===")
    print(" Train →", metrics["train"])
    print(" Test  →", metrics["test"], "\n")

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from itertools import product

def hyperparameter_search(model_cls, param_grid):
    records = []
    for params in product(*param_grid.values()):
        cfg = dict(zip(param_grid.keys(), params))

        if model_cls is GCNNet:
            model = model_cls(in_channels=data.num_features,
                              hidden_channels=cfg["hidden"]).to(device)
        else:
            model = model_cls(in_channels=data.num_features,
                              hidden_channels=cfg["hidden"],
                              heads=cfg["heads"]).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["wd"]
        )

        for epoch in range(1, 101):
            train_epoch(model, data, optimizer, criterion)

        # Evaluate
        metrics = evaluate(model, data)["test"]
        records.append({
            "model": model_cls.__name__,
            **cfg,
            "test_acc": metrics["acc"],
            "test_f1":  metrics["f1"],
            "test_auc": metrics["auc"]
        })

    return pd.DataFrame.from_records(records)

param_grid_common = {
    "hidden":      [32, 64],
    "lr":          [1e-3, 5e-3],
    "wd":          [5e-4, 1e-3]
}
param_grid_gat = {**param_grid_common, "heads": [2, 4]}

df_gcn   = hyperparameter_search(GCNNet, param_grid_common)
df_gat   = hyperparameter_search(GATNet, param_grid_gat)
df_gatv2 = hyperparameter_search(GATv2Net, param_grid_gat)

df_all = pd.concat([df_gcn, df_gat, df_gatv2], ignore_index=True)
df_all = df_all.sort_values("test_auc", ascending=False)

display(df_all.head(5))

"""## GraphSmote"""

!git clone https://github.com/TianxiangZhao/GraphSmote.git
# %cd GraphSmote

!find /content/GraphSmote -maxdepth 2 -type f | sed 's/\/content\/GraphSmote\///g'

!pip install ipdb

import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import ipdb
import copy

X = node_features_cls.cpu().numpy()
y = data.y.cpu().numpy()

features = data.x.cpu().numpy()
labels   = data.y.cpu().numpy()

!touch /content/GraphSmote-main

import sys
sys.path.append("/content/GraphSmote-main")

from utils import recon_upsample
from torch_geometric.utils import to_dense_adj

adj = to_dense_adj(data.edge_index)[0]

idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)

portion     = 1.0
im_class_num= num_classes

result = recon_upsample(
    embed        = data.x,
    labels       = data.y,
    idx_train    = idx_train,
    adj          = adj,
    portion      = 0.0,
    im_class_num = num_classes
)

for i, tensor in enumerate(result):
    print(f"result[{i}] shape:", tensor.shape)

print("Returned", len(result), "elements:", [type(r) for r in result])

raw = result[0]
loss_recon = raw.mean()
print("🤖 Recon loss (scalar):", loss_recon.item())

loss_recon = raw.sum()
print("🤖 Recon loss (sum):", loss_recon.item())

N = data.x.size(0)
synth = embed_new[N:]
orig_pad = embed_new[:N].detach()

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

N = data.x.size(0)
synth = embed_new[N:].cpu().numpy()
orig  = embed_new[:N].cpu().numpy()

sim_matrix = cosine_similarity(synth, orig)
best_sim   = sim_matrix.max(axis=1)

print("Mean max cosine sim (synth→orig):", best_sim.mean())

from collections import Counter

labels_all = result[1].cpu().tolist()
dist = Counter(labels_all)

print("Distribuição de classes após GraphSMOTE:")
for cls, cnt in sorted(dist.items()):
    print(f"  Classe {cls}: {cnt} nós")

from torch_geometric.utils import dense_to_sparse
from torch.optim import Adam

edge_index_new, _ = dense_to_sparse(adj_new)

labels_all = torch.tensor(labels_all, dtype=torch.long, device=embed_new.device)
data_new = Data(
    x=embed_new,
    edge_index=edge_index_new,
    y=labels_all
)

idx = torch.arange(data_new.num_nodes)
train_idx, test_idx = train_test_split(
    idx.cpu().numpy(),
    test_size=0.2,
    random_state=42,
    stratify=data_new.y.cpu().numpy()
)
train_mask = torch.zeros(data_new.num_nodes, dtype=torch.bool, device=embed_new.device)
test_mask  = torch.zeros_like(train_mask)
train_mask[train_idx] = True
test_mask[test_idx]   = True

data_new.train_mask = train_mask
data_new.test_mask  = test_mask

num_classes = int(data_new.y.max()) + 1

train_labels  = data_new.y[data_new.train_mask]
class_counts  = torch.bincount(train_labels)
class_weights = (1.0 / class_counts.float())
class_weights = (class_weights / class_weights.sum() * len(class_counts)).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

models = {
    "GCN":   GCNNet(data_new.num_features, 64),
    "GAT":   GATNet(data_new.num_features, 32, heads=4),
    "GATv2": GATv2Net(data_new.num_features, 32, heads=4),
}

def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs  = F.softmax(logits, dim=1)
    preds  = probs.argmax(dim=1)

    y_true = data.y.cpu().numpy()
    y_pred = preds.cpu().numpy()
    mask_train = data.train_mask.cpu().numpy()
    mask_test  = data.test_mask.cpu().numpy()

    results = {}
    for split, mask in [("train", mask_train), ("test", mask_test)]:
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1  = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        y_oh = torch.nn.functional.one_hot(data.y, num_classes=logits.size(1)).cpu().numpy()
        auc = roc_auc_score(y_oh[mask], probs[mask].cpu().numpy(), multi_class="ovr")
        results[split] = {"acc": acc, "f1": f1, "auc": auc}
    return results

results = {}
for name, model in models.items():
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    for epoch in range(1, 101):
        loss = train_epoch(model, data_new, optimizer, criterion)
        if epoch % 20 == 0:
            print(f"{name} Epoch {epoch:03d} Loss {loss:.4f}")

    metrics = evaluate(model, data_new)
    results[name] = metrics
    print(f"\n=== {name} Results ===")
    print(" Train →", metrics["train"])
    print(" Test  →", metrics["test"])
    print("-" * 40)

import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import torch
import numpy as np

y_true = data.y.cpu().numpy()
n_classes = int(data.y.max()) + 1

y_onehot = torch.nn.functional.one_hot(data.y, num_classes=n_classes).cpu().numpy()
test_mask = data.test_mask.cpu().numpy()

plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for (name, model), color in zip(models.items(), colors):
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    probs  = torch.softmax(logits, dim=1).cpu().detach().numpy()
    fpr, tpr, _ = roc_curve(y_onehot[test_mask].ravel(), probs[test_mask].ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-averaged ROC Curve (Test Set)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

for name, model in models.items():
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    preds  = logits.argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_true[test_mask], preds[test_mask], labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'Confusion Matrix - {name} (Test Set)')
    plt.show()

param_grid_common = {
    "hidden":       [32, 64, 128, 256],
    "lr":           [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "wd":           [1e-4, 5e-4, 1e-3, 5e-3]
}
param_grid_gat = {
    **param_grid_common,
    "heads":        [1, 2, 4, 8]
}

df_gcn   = hyperparameter_search(GCNNet,   param_grid_common)
df_gat   = hyperparameter_search(GATNet,   param_grid_gat)
df_gatv2 = hyperparameter_search(GATv2Net, param_grid_gat)

df_all = pd.concat([df_gcn, df_gat, df_gatv2], ignore_index=True)
df_all = df_all.sort_values("test_auc", ascending=False).reset_index(drop=True)

display(df_all.head(10))

HIDDEN_SIZE   = 64
LEARNING_RATE = 1e-2
WEIGHT_DECAY  = 5e-4
HEADS         = 1
EPOCHS        = 100
PRINT_EVERY   = 20

best_model = GATv2Net(
    in_channels=data_new.num_node_features,
    hidden_channels=HIDDEN_SIZE,
    heads=HEADS,
).to(device)

checkpoint_path = "/content/gatv2_best.pth"
torch.save(best_model.state_dict(), checkpoint_path)
print(f"Model weights saved to {checkpoint_path}")

import torch
import torch.nn.functional as F

def cascade_effect(model, data, node_idx: int, sigma: float = 0.05):
    model.eval()

    with torch.no_grad():
        base_logits = model(data.x, data.edge_index)
        base_probs  = F.softmax(base_logits, dim=1)[:, 1]

    x_pert = data.x.clone()
    noise  = torch.randn_like(x_pert[node_idx]) * sigma
    x_pert[node_idx] += noise

    with torch.no_grad():
        pert_logits = model(x_pert, data.edge_index)
        pert_probs  = F.softmax(pert_logits, dim=1)[:, 1]

    delta_self = (pert_probs[node_idx] - base_probs[node_idx]).item()

    src, dst = data.edge_index
    neighbour_ids = dst[src == node_idx]
    neighbour_deltas = (pert_probs[neighbour_ids] -
                        base_probs[neighbour_ids]).cpu()

    cascade_strength = neighbour_deltas.abs().mean().item() if len(neighbour_deltas) else 0.0

    return delta_self, neighbour_ids, neighbour_deltas, cascade_strength

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infer_model = GATv2Net(
    in_channels     = data_new.num_node_features,
    hidden_channels = 64,
    heads           = 1,

).to(device)
infer_model.load_state_dict(torch.load("/content/gatv2_best.pth", map_location=device))
infer_model.eval()

data_new = data_new.to(device)

node_id = 19
delta_self, nbrs, delta_nbrs, strength = cascade_effect(
    infer_model, data_new, node_id, sigma=1
)

print(f"Δ no nó {node_id}: {delta_self:+.4f}")
print(f"Vizinho(s): {nbrs.tolist()}")
for n, d in zip(nbrs.tolist(), delta_nbrs.tolist()):
    print(f"  Δ no vizinho {n}: {d:+.4f}")
print(f"Média |Δ| dos vizinhos: {strength:.4f}")

import torch
import numpy as np
from tqdm import tqdm

def rank_cascade_nodes(model, data, sigma=0.05, top_k=10):
    N = data.num_nodes
    strengths = np.zeros(N)

    for n in tqdm(range(N), desc="Scanning nodes"):
        _, _, _, strength = cascade_effect(model, data, n, sigma=sigma)
        strengths[n] = strength

    top_idx = np.argsort(-strengths)[:top_k]
    return top_idx.tolist(), strengths[top_idx].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_model.to(device).eval()
data_new = data_new.to(device)

top_nodes, top_strengths = rank_cascade_nodes(
    infer_model, data_new, sigma=0.05, top_k=10
)

print("\nTop 10 nós com maior cascade_strength:")
for rank, (n, s) in enumerate(zip(top_nodes, top_strengths), 1):
    print(f"{rank:>2}. nó {n:<6}  strength = {s:.4f}")