import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from collections import defaultdict

# ====================== code_to_graph Function ======================
def code_to_graph(code_str, label=0):
    """Convert C++ code to PyG graph with AST + CFG + basic PDG"""
    if not code_str or not code_str.strip():
        return None

    # Initialize parser (you need tree-sitter-cpp set up)
    global parser
    if 'parser' not in globals() or parser is None:
        from tree_sitter import Parser, Language
        # Adjust path if needed
        try:
            CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
            parser = Parser(CPP_LANGUAGE)
        except:
            print("Warning: Tree-sitter parser not initialized properly")
            return None

    tree = parser.parse(bytes(code_str, "utf-8"))
    
    nodes = []           # [type_id, num_children, is_statement]
    edges = []
    edge_types = []      # 0=AST, 1=CFG, 2=PDG
    
    statement_nodes = []
    var_defs = {}
    var_uses = defaultdict(list)

    def get_text(node):
        start = node.start_byte
        end = node.end_byte
        return code_str[start:end].decode("utf-8", errors="ignore") if isinstance(code_str, bytes) else code_str[start:end]

    def get_node_id(node_type):
        mapping = {
            "function_definition": 1, "if_statement": 2, "for_statement": 3,
            "while_statement": 4, "do_statement": 5, "return_statement": 6,
            "declaration": 7, "expression_statement": 8, "compound_statement": 9,
            "identifier": 10, "assignment_expression": 11, "binary_expression": 12,
            "update_expression": 13,
        }
        return mapping.get(node_type, 0)

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
            edge_types.append(0)
            edges.append([idx, parent_idx])
            edge_types.append(0)
        
        if is_stmt:
            statement_nodes.append((idx, node, get_text(node)))
        
        if node.type == "identifier":
            var_name = get_text(node).strip()
            if var_name:
                parent_type = node.parent.type if node.parent else ""
                if "assignment" in parent_type or "declaration" in parent_type:
                    var_defs.setdefault(var_name, []).append(idx)
                else:
                    var_uses[var_name].append(idx)
        
        for child in node.children:
            traverse(child, idx)

    traverse(tree.root_node)

    # CFG Edges
    for i in range(len(statement_nodes) - 1):
        curr_idx = statement_nodes[i][0]
        next_idx = statement_nodes[i + 1][0]
        edges.append([curr_idx, next_idx])
        edge_types.append(1)
        edges.append([next_idx, curr_idx])
        edge_types.append(1)

    # PDG Edges (Data Dependence)
    for var_name, defs in var_defs.items():
        uses = var_uses.get(var_name, [])
        for d_idx in defs:
            for u_idx in uses:
                if u_idx > d_idx:
                    edges.append([d_idx, u_idx])
                    edge_types.append(2)
                    edges.append([u_idx, d_idx])
                    edge_types.append(2)

    if len(nodes) < 2:
        return None

    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
    data.num_nodes = len(nodes)
    data.code = code_str

    return data


# ====================== CodeGNN Model ======================
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
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = CodeGNN(
        in_channels=checkpoint.get('in_channels', 3),
        hidden_channels=64,
        out_channels=checkpoint.get('out_channels', 128)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ CodeGNN model loaded successfully!")
    return model, device


def compute_code_similarity(model, code1: str, code2: str, device):
    """Compare two codes using CodeGNN"""
    try:
        g1 = code_to_graph(code1, label=0)
        g2 = code_to_graph(code2, label=0)
        
        if g1 is None or g2 is None:
            return "Error: Could not parse one or both codes"
        
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
