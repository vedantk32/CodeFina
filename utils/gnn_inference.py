import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data

# === FIXED IMPORT ===
try:
    from utils.code_to_graph import code_to_graph     # Absolute import (more reliable)
except ImportError:
    try:
        from .code_to_graph import code_to_graph      # Relative import fallback
    except ImportError:
        code_to_graph = None
        print("Warning: Could not import code_to_graph")

class CodeGNN(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


def load_gnn_model(model_path="models/code_gnn_model.pth"):
    device = torch.device('cpu')
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = CodeGNN(
        in_channels=checkpoint.get('in_channels', 3),
        hidden_channels=64,
        out_channels=checkpoint.get('out_channels', 128)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ CodeGNN model loaded successfully!")
    return model, device


def compute_code_similarity(model, code1: str, code2: str, device):
    """Compare two C++ codes using CodeGNN"""
    if code_to_graph is None:
        return "Error: code_to_graph function not found. Check file structure."

    try:
        g1 = code_to_graph(code1, label=0)
        g2 = code_to_graph(code2, label=0)
        
        if g1 is None or g2 is None:
            return "Error: Could not parse one or both code snippets"
        
        emb1 = get_embedding(model, g1, device)
        emb2 = get_embedding(model, g2, device)
        
        sim = F.cosine_similarity(emb1, emb2, dim=1).item()
        return round(sim * 100, 2)
        
    except Exception as e:
        return f"Comparison Error: {str(e)}"


def get_embedding(model, graph_data, device):
    model.eval()
    with torch.no_grad():
        graph_data = graph_data.to(device)
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
        emb = model(graph_data.x, graph_data.edge_index, batch)
        return emb
