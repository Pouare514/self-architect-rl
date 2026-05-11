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

class AddModule(nn.Module):
    """Element-wise addition of multiple inputs. Used for skip/residual connections."""
    is_multi_input = True

    def forward(self, inputs):
        result = inputs[0]
        for inp in inputs[1:]:
            result = result + inp
        return result


class DAGModule(nn.Module):
    def __init__(self, modules: nn.ModuleDict, edges: list, input_node_id="input"):
        super().__init__()
        self.layers = modules
        self.edges = edges
        self.input_node_id = input_node_id
        self._topo_order = None

    def _topo_sort(self):
        """Kahn's algorithm for topological ordering of the DAG."""
        from collections import defaultdict, deque

        in_degree = defaultdict(int)
        adj = defaultdict(list)
        all_nodes = {self.input_node_id}

        for edge in self.edges:
            all_nodes.add(edge.from_id)
            all_nodes.add(edge.to_id)
            adj[edge.from_id].append(edge.to_id)
            in_degree[edge.to_id] += 1

        queue = deque(n for n in all_nodes if in_degree[n] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def _build_incoming_map(self):
        from collections import defaultdict
        incoming = defaultdict(list)
        for edge in self.edges:
            incoming[edge.to_id].append(edge.from_id)
        return incoming

    def forward(self, x):
        outputs = {self.input_node_id: x}

        # Compute topological order (cached after first call)
        if self._topo_order is None:
            self._topo_order = self._topo_sort()
            self._incoming = self._build_incoming_map()

        for node_id in self._topo_order:
            if node_id == self.input_node_id:
                continue

            # Collect all inputs from incoming edges
            inputs = [outputs[src] for src in self._incoming[node_id] if src in outputs]
            if not inputs:
                continue

            if node_id in self.layers:
                layer = self.layers[node_id]
                if getattr(layer, 'is_multi_input', False):
                    outputs[node_id] = layer(inputs)
                else:
                    outputs[node_id] = layer(inputs[0])
            else:
                # Pass-through (non-layer node)
                outputs[node_id] = inputs[0]

        # Return the last computed node in topological order
        return outputs[self._topo_order[-1]]

    def invalidate_topo_cache(self):
        """Call after modifying edges to force recomputation of topological order."""
        self._topo_order = None
        self._incoming = None


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
        "add": AddModule,
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

