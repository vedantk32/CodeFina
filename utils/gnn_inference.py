from tree_sitter import Language, Parser
import tree_sitter_cpp
import torch
from torch_geometric.data import Data
from pathlib import Path
from torch_geometric.nn import SAGEConv, global_mean_pool

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

        return torch.nn.functional.normalize(x, p=2, dim=1)


def load_gnn_model(model_path="models/code_gnn_model.pth"):
    device = torch.device('cpu')  # Safe for Streamlit Cloud
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    model = CodeGNN(
        in_channels=checkpoint.get('in_channels', 3),
        hidden_channels=64,
        out_channels=checkpoint.get('out_channels', 128)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device


def get_embedding(model, graph_data, device):
    model.eval()
    with torch.no_grad():
        graph_data = graph_data.to(device)
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
        emb = model(graph_data.x, graph_data.edge_index, batch)
        return emb


def compute_code_similarity(model, code1: str, code2: str, device):
    """Compare two C++ codes using CodeGNN"""
    try:
        # TODO: Replace with your actual code_to_graph function that includes AST+CFG+PDG
        # For now using placeholder - you need to implement proper graph creation
        g1 = placeholder_code_to_graph(code1)
        g2 = placeholder_code_to_graph(code2)
        
        emb1 = get_embedding(model, g1, device)
        emb2 = get_embedding(model, g2, device)
        
        sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
        return round(sim * 100, 2)
    except Exception as e:
        return f"Error: {str(e)}"


# Placeholder - Replace this with your real tree-sitter based code_to_graph

def code_to_graph(code_str, label):
    tree = parser.parse(bytes(code_str, "utf8"))
    nodes = []          # list of features [type_id, num_children, is_statement]
    edges = []          # list of [src, dst]
    edge_types = []     # 0=AST, 1=Control Flow, 2=Data Dependence

    statement_nodes = []  # (idx, tree_node, text)
    var_defs = {}         # var_name -> list of definition node indices (simple tracking)
    var_uses = defaultdict(list)

    def get_text(node):
     if isinstance(code_str, bytes):          # code_str is the global/outer variable holding the source
        text = code_str[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
     else:
        text = code_str[node.start_byte:node.end_byte]   # already str, just slice
     return text

    def traverse(node, parent_idx=None):
        idx = len(nodes)

        is_stmt = 1 if node.type in [
            "if_statement", "for_statement", "while_statement", "do_statement",
            "expression_statement", "return_statement", "compound_statement"
        ] else 0

        feature = [get_node_id(node.type), len(node.children), is_stmt]
        nodes.append(feature)

        if parent_idx is not None:
            edges.append([parent_idx, idx])
            edge_types.append(0)  # AST edge
            edges.append([idx, parent_idx])
            edge_types.append(0)

        # Track statements for CFG
        if is_stmt:
            statement_nodes.append((idx, node, get_text(node)))

        # Simple data flow tracking (variable defs/uses)
        if node.type == "identifier":
            var_name = get_text(node).strip()
            if var_name:
                # Heuristic: previous assignment = def, else use
                # In real PDG you'd do proper reaching definitions
                if parent_idx is not None and "assignment" in str(node.parent.type) if node.parent else False:
                    var_defs.setdefault(var_name, []).append(idx)
                else:
                    var_uses[var_name].append(idx)

        for child in node.children:
            traverse(child, idx)

    traverse(tree.root_node)

    # ==================== Control Flow (CFG) Edges ====================
    for i in range(len(statement_nodes) - 1):
        curr_idx, curr_node, _ = statement_nodes[i]
        next_idx, _, _ = statement_nodes[i + 1]

        # Sequential flow
        edges.append([curr_idx, next_idx])
        edge_types.append(1)  # Control Flow
        edges.append([next_idx, curr_idx])
        edge_types.append(1)

        # Branching for if
        if curr_node.type == "if_statement":
            for child in curr_node.children:
                if child.type in ["compound_statement", "expression_statement"]:
                    edges.append([curr_idx, curr_idx])  # placeholder for branch start
                    edge_types.append(1)

        # Loop back-edges
        if curr_node.type in ["for_statement", "while_statement", "do_statement"]:
            edges.append([next_idx, curr_idx])
            edge_types.append(1)

    # ==================== Data Dependence Edges ====================
    for var_name, defs in var_defs.items():
        uses = var_uses.get(var_name, [])
        for d_idx in defs:
            for u_idx in uses:
                if u_idx > d_idx:  # forward data flow only
                    edges.append([d_idx, u_idx])
                    edge_types.append(2)  # Data Dependence
                    edges.append([u_idx, d_idx])  # bidirectional for GNN
                    edge_types.append(2)

    # Build PyG Data
    if len(nodes) < 2:
        return None

    x = torch.tensor(nodes, dtype=torch.float)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
    data.edge_type = edge_type  # heterogeneous edge types
    data.code = code_str
    data.num_nodes = len(nodes)

    return data
