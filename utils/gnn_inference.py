import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data

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
    device = torch.device('cpu')   # Important for Streamlit Cloud
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {model_path}. Make sure the model is in the 'models/' folder.")

    model = CodeGNN(
        in_channels=checkpoint.get('in_channels', 3),
        hidden_channels=64,
        out_channels=checkpoint.get('out_channels', 128)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ CodeGNN model loaded successfully from {model_path}")
    return model, device


def get_embedding(model, graph_data, device):
    model.eval()
    with torch.no_grad():
        graph_data = graph_data.to(device)
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
        emb = model(graph_data.x, graph_data.edge_index, batch)
        return emb


# Placeholder - Replace this with your real code_to_graph later
def code_to_graph(code_str, label):
    if not code_str or not code_str.strip():
        return None
    
    # Parse the code
    tree = parser.parse(bytes(code_str, "utf-8"))
    
    nodes = []           # list of [type_id, num_children, is_statement]
    edges = []           # list of [src, dst]
    edge_types = []      # 0=AST, 1=CFG, 2=PDG
    
    statement_nodes = [] # (idx, node, text)
    var_defs = {}        # var_name -> list of def indices
    var_uses = defaultdict(list)

    def get_text(node):
        start = node.start_byte
        end = node.end_byte
        return code_str[start:end].decode("utf-8", errors="ignore") if isinstance(code_str, bytes) else \
               code_str[start:end]

    def get_node_id(node_type):
        # Simple consistent mapping (you can improve this later)
        common_types = {
            "function_definition": 1, "if_statement": 2, "for_statement": 3,
            "while_statement": 4, "return_statement": 5, "declaration": 6,
            "expression_statement": 7, "compound_statement": 8, "identifier": 9,
            "assignment_expression": 10, "binary_expression": 11
        }
        return common_types.get(node_type, 0)   # 0 = unknown

    def traverse(node, parent_idx=None):
        idx = len(nodes)
        
        is_stmt = 1 if node.type in [
            "if_statement", "for_statement", "while_statement", "do_statement",
            "expression_statement", "return_statement", "compound_statement"
        ] else 0
        
        feature = [
            get_node_id(node.type),      # f1: Node Type
            len(node.children),          # f2: Number of children
            is_stmt                      # f3: Is statement?
        ]
        
        nodes.append(feature)
        
        if parent_idx is not None:
            edges.append([parent_idx, idx])
            edge_types.append(0)  # AST
            edges.append([idx, parent_idx])
            edge_types.append(0)
        
        if is_stmt:
            statement_nodes.append((idx, node, get_text(node)))
        
        # Simple data flow tracking
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

    # ==================== CFG Edges ====================
    for i in range(len(statement_nodes) - 1):
        curr_idx, _, _ = statement_nodes[i]
        next_idx, _, _ = statement_nodes[i + 1]
        edges.append([curr_idx, next_idx])
        edge_types.append(1)
        edges.append([next_idx, curr_idx])
        edge_types.append(1)

    # ==================== PDG (Data Dependence) Edges ====================
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
