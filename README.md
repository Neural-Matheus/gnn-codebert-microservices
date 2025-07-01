# Graph Neural Networks for Failure Propagation in Microservice Architectures

This project contains code and documentation to analyze call graphs and train Graph Neural Network models to study failure propagation in microservice architectures, using the Saleor repository as a case study.

## 📁 Project Structure

- **graph.py**  
  Script to clone a repository, extract functions, build the call graph, compute graph metrics, and analyze cyclomatic complexity.

- **model.py**  
  Script to load the serialized graph, generate embeddings, train and evaluate three GNN models (GCN, GAT, GATv2), and apply oversampling with GraphSMOTE.

- **article.pdf**  
  Academic paper detailing the proposal, methodology, results, and discussion.

## 🛠️ Installation and Setup

1. **Create a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Automatic setup**  
   ```bash
   bash setup.sh
   ```

4. **Generate the call graph**  
   ```bash
   python graph.py --repo-url https://github.com/mirumee/saleor --output call_graph.pkl
   ```
   This command produces `call_graph.pkl` and a 3D visualization in `call_graph.html`.


## 🚀 Usage

### 1. Call Graph Analysis (`graph.py`)

```bash
python graph.py \
  --repo-url <REPO_URL> \
  --clone-dir saleor_src \
  --output-graph call_graph.pkl \
  --html-output call_graph.html
```

- `--repo-url`: Git repository URL to analyze
- `--clone-dir`: Directory to clone the repository into
- `--output-graph`: Output pickle file for the call graph
- `--html-output`: HTML file for the 3D visualization

### 2. Model Training (`model.py`)

```bash
python model.py \
  --input-graph call_graph_with_label.pkl \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.005 \
  --output-dir results/
```

- `--input-graph`: Labeled graph pickle indicating failure/non-failure nodes
- `--epochs`: Maximum number of training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--output-dir`: Directory to save checkpoints and plots


## 📊 Expected Results

- **Metrics plots**: AUROC, AUPRC, confusion matrices saved in `results/`
- **Model checkpoints**: `.pth` files for GCN, GAT, and GATv2
- **Report**: See `article.pdf` for detailed analysis and discussion

---
