# -*- coding: utf-8 -*-
import subprocess
import os

repo_url = "https://github.com/saleor/saleor.git"
target_dir = "saleor"

if not os.path.exists(target_dir):
    subprocess.run(["git", "clone", repo_url, target_dir], check=True)

os.chdir(target_dir)
print(f"✅ Cloned and moved into {target_dir}")

import os
import glob
import ast
import textwrap

root      = os.getcwd()
code_root = root

packages = [
    name for name in os.listdir(code_root)
    if os.path.isdir(os.path.join(code_root, name))
    and os.path.isfile(os.path.join(code_root, name, "__init__.py"))
]

service_snippets = {}

for pkg in packages:
    pkg_path = os.path.join(code_root, pkg)
    snippets = {}
    for py_file in glob.glob(f"{pkg_path}/**/*.py", recursive=True):
        with open(py_file, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source)
        except (SyntaxError, IndentationError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start, end = node.lineno, node.end_lineno
                lines = source.splitlines()
                snippet = "\n".join(lines[start-1 : end])
                key = f"{pkg}:{node.name}"
                snippets[key] = textwrap.dedent(snippet)

    service_snippets[pkg] = snippets

print(
    f"✅ Extracted functions from {len(packages)} packages, totaling "
    f"{sum(len(v) for v in service_snippets.values())} snippets."
)

import networkx as nx
call_graph = nx.DiGraph()

for pkg, snippets in service_snippets.items():
    for node_id in snippets.keys():
        call_graph.add_node(node_id)

print(f"✅ Added {call_graph.number_of_nodes()} nodes to the graph")

import ast
import textwrap

for pkg, snippets in service_snippets.items():
    for node_id, code in snippets.items():
        caller = node_id
        snippet = textwrap.dedent(code)
        try:
            tree = ast.parse(snippet)
        except IndentationError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            if isinstance(func, ast.Name):
                callee = func.id
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                callee = func.attr
            else:
                continue

            callee_full = f"{pkg}:{callee}"
            if callee_full != caller and call_graph.has_node(callee_full):
                call_graph.add_edge(caller, callee_full)

print(f"✅ Built edges, total edges: {call_graph.number_of_edges()}")

print("Nodes:")
for node in call_graph.nodes():
    print(node)

print("\nEdges:")
for src, dst in call_graph.edges():
    print(f"{src} -> {dst}")

import textwrap
import re


for src, dst in call_graph.edges():
    pkg_src, _    = src.split(":", 1)
    _,     fn_dst = dst.split(":", 1)
    snippet = service_snippets[pkg_src].get(src)
    if not snippet:
        continue

    print(f"\n📌 Edge: {src} → {dst}")
    print(f"--- Source node `{src}` ---")
    code = textwrap.dedent(snippet)
    print(code)

    pattern = rf"\b{re.escape(fn_dst)}\s*\("
    matches = list(re.finditer(pattern, code))
    if matches:
        for m in matches:
            line_no = code[: m.start()].count("\n") + 1
            line    = code.splitlines()[line_no - 1]
            print(f"    -> calls `{fn_dst}` at line {line_no}: {line.strip()}")
    else:
        print(f"    ⚠️ No explicit call to `{fn_dst}` found")

"""## Limpeza do grafo, remover funções de teste"""

import networkx as nx

original_nodes = list(call_graph.nodes())
print(f"📌 Original nodes: {len(original_nodes)}")

print("Sample original nodes:", original_nodes[:10], "...")

nodes_before_iso = list(call_graph.nodes())
edges_before_iso = list(call_graph.edges())
print(f"📌 Nodes before dropping isolates: {len(nodes_before_iso)}")
print(f"📌 Edges before dropping isolates: {len(edges_before_iso)}")

isolates = list(nx.isolates(call_graph))
print(f"🧹 Isolated nodes found: {len(isolates)}")
call_graph.remove_nodes_from(isolates)

nodes_after_iso = list(call_graph.nodes())
edges_after_iso = list(call_graph.edges())
print(f"✅ Nodes after dropping isolates: {len(nodes_after_iso)}")
print(f"✅ Edges after dropping isolates: {len(edges_after_iso)}")
print(f"🗑️ Total nodes removed: {len(nodes_before_iso) - len(nodes_after_iso)}")

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

pos_3d = nx.spring_layout(call_graph, dim=3, seed=42)
nodes = list(call_graph.nodes())

palette = px.colors.qualitative.Plotly
node_color_map = {node: palette[i % len(palette)] for i, node in enumerate(nodes)}

edge_traces = []
for src, dst in call_graph.edges():
    x0, y0, z0 = pos_3d[src]
    x1, y1, z1 = pos_3d[dst]
    edge_traces.append(
        go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(width=2, color=node_color_map[src]),
            hoverinfo='none',
            showlegend=False
        )
    )

node_x, node_y, node_z = [], [], []
hover_text = []
for node in nodes:
    x, y, z = pos_3d[node]
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)
    hover_text.append(node.split(":",1)[1])

node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    marker=dict(size=8, color=[node_color_map[n] for n in nodes],
                line=dict(width=1, color='black')),
    hoverinfo='text',
    hovertext=hover_text
)

fig = go.Figure(data=edge_traces + [node_trace])
fig.update_layout(
    title="3D Call Graph (hover names)",
    scene=dict(
        xaxis=dict(showbackground=False, visible=False),
        yaxis=dict(showbackground=False, visible=False),
        zaxis=dict(showbackground=False, visible=False)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    plot_bgcolor='white'
)

fig.show()

import textwrap
import re


for src, dst in call_graph.edges():
    svc_src, fn_src = src.split(":", 1)
    svc_dst, fn_dst = dst.split(":", 1)
    snippet = service_snippets[svc_src].get(fn_src)
    if not snippet:
        continue

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

import networkx as nx
import statistics

n_nodes = call_graph.number_of_nodes()
n_edges = call_graph.number_of_edges()
print(f"📊 Nodes: {n_nodes}")
print(f"📊 Edges: {n_edges}")

"""# Análise de criticidade de código"""

!pip install radon

for node in call_graph.nodes():
    svc, fname = node.split(":", 1)
    snippet = service_snippets.get(svc, {}).get(node)
    if snippet:
        call_graph.nodes[node]["code"] = snippet

import textwrap
from radon.complexity import cc_visit, cc_rank

some_node = next(iter(call_graph.nodes()))
print(f"📌 Analyzing node: {some_node}")

code = textwrap.dedent(call_graph.nodes[some_node]["code"])
blocks = cc_visit(code)
for block in blocks:
    comp   = block.complexity
    letter = cc_rank(comp)
    print(f"- Function `{block.name}`: complexity={comp}, rank={letter}")

from radon.complexity import cc_visit, cc_rank
import textwrap
import pandas as pd

complexity_results = {}

for node in call_graph.nodes():
    code = call_graph.nodes[node].get("code", "")
    code = textwrap.dedent(code)

    try:
        blocks = cc_visit(code)
        if blocks:
            complexity = blocks[0].complexity
        else:
            complexity = 0
    except Exception:
        complexity = 0

    rank = cc_rank(complexity)

    complexity_results[node] = {"complexity": complexity, "rank": rank}
    print(f"{node}: complexity={complexity}, rank={rank}")

df = pd.DataFrame.from_dict(complexity_results, orient="index")
df.index.name = "node"
display(df)

rank_counts = df["rank"].value_counts().sort_index()
print("\nNode counts by complexity rank:")
for rank, count in rank_counts.items():
    print(f"  {rank}: {count}")

import pickle
import textwrap

rank_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5
}

numeric_results = {}
for node in call_graph.nodes():
    code = textwrap.dedent(call_graph.nodes[node].get("code", ""))
    try:
        blocks = cc_visit(code)
        comp = blocks[0].complexity if blocks else 0
    except Exception:
        comp = 0
    letter = cc_rank(comp)
    num_label = rank_mapping.get(letter, 0)
    numeric_results[node] = {"complexity": comp, "label": num_label}

for node, vals in numeric_results.items():
    call_graph.nodes[node]["label"] = vals["label"]

import networkx as nx
import statistics

n_nodes = call_graph.number_of_nodes()
n_edges = call_graph.number_of_edges()
print(f"📊 Nodes: {n_nodes}")
print(f"📊 Edges: {n_edges}")

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

import pickle

filepath = "/content/call_graph_with_label.pkl"
with open(filepath, "wb") as f:
    pickle.dump(call_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✅ Graph saved with labels to {filepath}")

with open("/content/call_graph_with_label.pkl", "rb") as f:
    cg = pickle.load(f)

print(cg.number_of_nodes(), "nodes;", cg.number_of_edges(), "edges")
node = next(iter(cg.nodes()))
print("Example:", node)
print(" Label:", cg.nodes[node]["label"])
print(" Code snippet (first 100 chars):", cg.nodes[node]["code"])

from collections import Counter
import pandas as pd

labels = [data["label"] for _, data in cg.nodes(data=True)]
label_counts = Counter(labels)
print("Label counts:", label_counts)

print("Unique labels:", sorted(label_counts.keys()))

for lbl in sorted(label_counts.keys()):
    samples = [
        node
        for node, data in cg.nodes(data=True)
        if data["label"] == lbl
    ][:5]
    print(f"Label {lbl} — sample nodes: {samples}")

df = pd.DataFrame([
    {
        "node": node,
        "label": data["label"],
        "snippet_preview": data["code"][:50].replace("\n"," ") + "..."
    }
    for node, data in cg.nodes(data=True)
])
display(df.head(10))