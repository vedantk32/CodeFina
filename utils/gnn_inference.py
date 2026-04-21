import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from collections import defaultdict
from pathlib import Path
import pickle

# -----------------------------
# TREE-SITTER SETUP
# -----------------------------
from tree_sitter import Parser
import tree_sitter_cpp

parser = Parser()
parser.set_language(tree_sitter_cpp.language())

# -----------------------------
# LOAD NODE VOCAB (IMPORTANT)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
VOCAB_PATH = BASE_DIR / "models" / "node_vocab.pkl"

if VOCAB_PATH.exists():
    with open(VOCAB_PATH, "rb") as f:
        node_vocab = pickle.load(f)
else:
    node_vocab = {}  # fallback (not recommended)

def get_node_id(node_type):
    if node_type not in node_vocab:
        node_vocab[node_type] = len(node_vocab)
    return node_vocab[node_type]

# -----------------------------
# MODEL
# -----------------------------
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

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_gnn_model():
    device = torch.device("cpu")

    model_path = BASE_DIR / "models" / "code_gnn_model.pth"
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint.get("config", {
        "in_channels": 3,
        "hidden_channels": 64,
        "out_channels": 128
    })

    model = CodeGNN(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_channels"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device

# -----------------------------
# GRAPH CREATION (AST + CFG + PDG)
# -----------------------------
def code_to_graph(code_str, label=0):

    tree = parser.parse(bytes(code_str, "utf8"))

    nodes = []
    edges = []
    edge_types = []

    statement_nodes = []
    var_defs = defaultdict(list)
    var_uses = defaultdict(list)

    def get_text(node):
        return code_str[node.start_byte:node.end_byte]

    def traverse(node, parent_idx=None):
        idx = len(nodes)

        is_stmt = 1 if node.type in [
            "if_statement", "for_statement", "while_statement",
            "expression_statement", "return_statement", "compound_statement"
        ] else 0

        feature = [get_node_id(node.type), len(node.children), is_stmt]
        nodes.append(feature)

        # AST edges
        if parent_idx is not None:
            edges.append([parent_idx, idx])
            edge_types.append(0)
            edges.append([idx, parent_idx])
            edge_types.append(0)

        if is_stmt:
            statement_nodes.append((idx, node))

        # Variable tracking (simplified PDG)
        if node.type == "identifier":
            name = get_text(node).strip()
            if name:
                var_uses[name].append(idx)

        for child in node.children:
            traverse(child, idx)

    traverse(tree.root_node)

    # ---------------- CFG ----------------
    for i in range(len(statement_nodes) - 1):
        a_idx, a_node = statement_nodes[i]
        b_idx, _ = statement_nodes[i + 1]

        edges.append([a_idx, b_idx])
        edge_types.append(1)
        edges.append([b_idx, a_idx])
        edge_types.append(1)

    # ---------------- PDG (simple) ----------------
    for var, uses in var_uses.items():
        for i in range(len(uses) - 1):
            edges.append([uses[i], uses[i + 1]])
            edge_types.append(2)
            edges.append([uses[i + 1], uses[i]])
            edge_types.append(2)

    if len(nodes) < 2:
        return None

    x = torch.tensor(nodes, dtype=torch.float)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(nodes)

    return data

# -----------------------------
# EMBEDDING
# -----------------------------
def get_embedding(model, graph, device):
    with torch.no_grad():
        graph = graph.to(device)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        return model(graph.x, graph.edge_index, batch)

# -----------------------------
# SIMILARITY
# -----------------------------
def compute_code_similarity(model, code1, code2, device):
    try:
        g1 = code_to_graph(code1)
        g2 = code_to_graph(code2)

        if g1 is None or g2 is None:
            return "Error: Invalid code input"

        emb1 = get_embedding(model, g1, device)
        emb2 = get_embedding(model, g2, device)

        sim = F.cosine_similarity(emb1, emb2).item()

        return round(sim * 100, 2)

    except Exception as e:
        return f"Error: {str(e)}"
