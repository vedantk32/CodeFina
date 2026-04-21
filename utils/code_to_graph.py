import torch
from torch_geometric.data import Data
from collections import defaultdict
from tree_sitter import Parser

# Global parser (initialize once)
parser = None

def init_parser():
    global parser
    if parser is None:
        # Make sure you have tree-sitter-cpp installed and built
        # You can run this once in your environment:
        # Language.build_library('build/my-languages.so', ['vendor/tree-sitter-cpp'])
        from tree_sitter import Language
        CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
        parser = Parser(CPP_LANGUAGE)
    return parser


def code_to_graph(code_str, label=0):
    """
    Convert C++ code to PyG graph with AST + CFG + basic PDG
    Node features: [node_type_id, num_children, is_statement]
    """
    if not code_str or not code_str.strip():
        return None

    parser = init_parser()
    
    # Parse the code
    tree = parser.parse(bytes(code_str, "utf-8"))
    
    nodes = []           # [type_id, num_children, is_statement]
    edges = []           # [src, dst]
    edge_types = []      # 0=AST, 1=CFG, 2=PDG
    
    statement_nodes = [] # (idx, node, text)
    var_defs = {}        # var_name -> list of definition indices
    var_uses = defaultdict(list)

    def get_text(node):
        start = node.start_byte
        end = node.end_byte
        return code_str[start:end].decode("utf-8", errors="ignore") if isinstance(code_str, bytes) else \
               code_str[start:end]

    def get_node_id(node_type: str) -> int:
        """Simple node type encoding"""
        mapping = {
            "function_definition": 1,
            "if_statement": 2,
            "for_statement": 3,
            "while_statement": 4,
            "do_statement": 5,
            "return_statement": 6,
            "declaration": 7,
            "expression_statement": 8,
            "compound_statement": 9,
            "identifier": 10,
            "assignment_expression": 11,
            "binary_expression": 12,
            "update_expression": 13,
        }
        return mapping.get(node_type, 0)   # 0 = unknown

    def traverse(node, parent_idx=None):
        idx = len(nodes)
        
        is_stmt = 1 if node.type in [
            "if_statement", "for_statement", "while_statement", "do_statement",
            "expression_statement", "return_statement", "compound_statement"
        ] else 0
        
        # Feature vector: 3 dimensions
        feature = [
            get_node_id(node.type),      # f1: Node Type ID
            len(node.children),          # f2: Number of children
            is_stmt                      # f3: Is Statement?
        ]
        
        nodes.append(feature)
        
        # Add AST edges (bidirectional)
        if parent_idx is not None:
            edges.append([parent_idx, idx])
            edge_types.append(0)
            edges.append([idx, parent_idx])
            edge_types.append(0)
        
        # Track statements for CFG
        if is_stmt:
            statement_nodes.append((idx, node, get_text(node)))
        
        # Simple variable definition / use tracking for PDG
        if node.type == "identifier":
            var_name = get_text(node).strip()
            if var_name:
                parent = node.parent
                parent_type = parent.type if parent else ""
                if "assignment" in parent_type or "declaration" in parent_type:
                    var_defs.setdefault(var_name, []).append(idx)
                else:
                    var_uses[var_name].append(idx)
        
        # Recurse
        for child in node.children:
            traverse(child, idx)

    # Start traversal
    traverse(tree.root_node)

    # ====================== CFG Edges ======================
    for i in range(len(statement_nodes) - 1):
        curr_idx, _, _ = statement_nodes[i]
        next_idx, _, _ = statement_nodes[i + 1]
        edges.append([curr_idx, next_idx])
        edge_types.append(1)   # Control Flow
        edges.append([next_idx, curr_idx])
        edge_types.append(1)

    # ====================== PDG Edges (Data Dependence) ======================
    for var_name, defs in var_defs.items():
        uses = var_uses.get(var_name, [])
        for d_idx in defs:
            for u_idx in uses:
                if u_idx > d_idx:          # forward data flow
                    edges.append([d_idx, u_idx])
                    edge_types.append(2)   # Data Dependence
                    edges.append([u_idx, d_idx])
                    edge_types.append(2)

    # Build PyTorch Geometric Data object
    if len(nodes) < 2:
        return None

    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
    data.num_nodes = len(nodes)
    data.code = code_str

    return data
