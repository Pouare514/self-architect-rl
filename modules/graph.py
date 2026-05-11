import math
import torch
import torch.nn as nn

class NodeDef:
    def __init__(self, node_id, layer_type, config):
        self.id = node_id
        self.layer_type = layer_type
        self.config = config

class EdgeDef:
    def __init__(self, from_id, to_id, connection_type="sequential"):
        self.from_id = from_id
        self.to_id = to_id
        self.connection_type = connection_type

def get_channel_importances(module, norm="l1"):
    """Return importance scores for each output channel of a Linear or Conv2d layer."""
    if isinstance(module, nn.Linear):
        weights = module.weight.data
        if norm == "l1":
            return weights.abs().sum(dim=1)
        elif norm == "l2":
            return torch.sqrt((weights ** 2).sum(dim=1) + 1e-12)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    elif isinstance(module, nn.Conv2d):
        weights = module.weight.data
        if norm == "l1":
            return weights.abs().sum(dim=(1, 2, 3))
        elif norm == "l2":
            return torch.sqrt((weights ** 2).sum(dim=(1, 2, 3)) + 1e-12)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    else:
        raise ValueError("get_channel_importances only supports nn.Linear and nn.Conv2d")

class GraphDef:
    """
    Représente la topologie de l'architecture de l'agent.
    Le configurateur agit sur cet objet pour modifier l'architecture.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []
        
    def add_node(self, node_id, layer_type, config=None):
        if config is None:
            config = {}
        self.nodes[node_id] = NodeDef(node_id, layer_type, config)
        return self
        
    def add_edge(self, from_id, to_id, connection_type="sequential"):
        self.edges.append(EdgeDef(from_id, to_id, connection_type))
        return self

class DAGModule(nn.Module):
    def __init__(self, modules: nn.ModuleDict, edges: list, input_node_id="input"):
        super().__init__()
        self.layers = modules
        self.edges = edges
        self.input_node_id = input_node_id
        
    def forward(self, x):
        outputs = {self.input_node_id: x}
        # Execution naive selon l'ordre des edges (doit être trié topologiquement)
        for edge in self.edges:
            if edge.from_id in outputs:
                in_val = outputs[edge.from_id]
                # Si le noeud cible n'a pas encore été calculé
                if edge.to_id not in outputs:
                    if edge.to_id in self.layers:
                        outputs[edge.to_id] = self.layers[str(edge.to_id)](in_val)
                    else:
                        outputs[edge.to_id] = in_val # Si ce n'est pas une couche, pass-through
        
        # Retourne la dernière valeur calculée
        return outputs[self.edges[-1].to_id]

class ModuleBuilder:
    """
    Convertit un GraphDef en modules PyTorch exécutables.
    """
    LAYER_REGISTRY = {
        "linear": nn.Linear,
        "conv2d": nn.Conv2d,
        "relu": nn.ReLU,
        "flatten": nn.Flatten,
        "layer_norm": nn.LayerNorm,
    }

    @classmethod
    def register_layer(cls, name, layer_class):
        cls.LAYER_REGISTRY[name] = layer_class

    @staticmethod
    def build(graph_def: GraphDef, input_node_id="input") -> nn.Module:
        modules = nn.ModuleDict()
        for node_id, node in graph_def.nodes.items():
            if node.layer_type in ModuleBuilder.LAYER_REGISTRY:
                layer_class = ModuleBuilder.LAYER_REGISTRY[node.layer_type]
                modules[str(node_id)] = layer_class(**node.config)
            else:
                raise ValueError(f"Unknown layer type: {node.layer_type}")
                
        return DAGModule(modules, graph_def.edges, input_node_id)
