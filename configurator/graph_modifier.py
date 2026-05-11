import math
import torch
import torch.nn as nn
from modules.graph import GraphDef, get_channel_importances

class GraphModifier:
    """
    Reçoit les actions du configurateur et les applique sur le GraphDef de l'agent.
    Gère la préservation des poids lors des modifications (héritage, adaptation de dimensions).
    """
    POLICY_ADD_LAYER = 1
    VALUE_ADD_LAYER = 2
    POLICY_WIDEN = 3
    VALUE_WIDEN = 4
    POLICY_PRUNE_LAYER = 5
    VALUE_PRUNE_LAYER = 6
    PHASE_PRUNE = "prune"
    PHASE_GROW = "grow"

    def __init__(self, prune_ratio=0.25, min_hidden_dim=64):
        self.mutation_count = 0
        self.prune_ratio = prune_ratio
        self.min_hidden_dim = min_hidden_dim
        self.phase = None
        self.phase_cycle_length = 4
        self.phase_prune_length = 2
        
    def set_meta_phase(self, meta_step, cycle_length=4, prune_phase_length=2):
        self.phase_cycle_length = cycle_length
        self.phase_prune_length = prune_phase_length
        phase_index = meta_step % cycle_length
        self.phase = self.PHASE_PRUNE if phase_index < prune_phase_length else self.PHASE_GROW
        return self.phase

    def _find_next_layer_with_in_features(self, graph, start_node_id):
        current = start_node_id
        while True:
            next_node_id = None
            for e in graph.edges:
                if e.from_id == current:
                    next_node_id = e.to_id
                    break
            if not next_node_id:
                return None
            next_node = graph.nodes.get(next_node_id)
            if next_node and 'in_features' in next_node.config:
                return next_node
            current = next_node_id

    def _prune_hidden_layer(self, graph, target_node_id, label):
        if target_node_id not in graph.nodes:
            return False
        node = graph.nodes[target_node_id]
        out_features = node.config.get('out_features')
        if out_features is None:
            return False

        prune_count = max(int(math.ceil(out_features * self.prune_ratio)), 16)
        new_dim = max(self.min_hidden_dim, out_features - prune_count)
        if new_dim >= out_features:
            return False

        node.config['out_features'] = new_dim
        next_node = self._find_next_layer_with_in_features(graph, target_node_id)
        if next_node:
            next_node.config['in_features'] = new_dim

        self.mutation_count += 1
        print(f"[Architect] Mutated {label}: Pruned {target_node_id} from {out_features} to {new_dim}")
        return True

    def apply_action(self, agent_graphs, action_idx):
        mutated = False

        if self.phase == self.PHASE_PRUNE and action_idx in (self.POLICY_ADD_LAYER, self.VALUE_ADD_LAYER, self.POLICY_WIDEN, self.VALUE_WIDEN):
            print(f"[Architect] Deferred grow action {action_idx} during prune phase")
            return False
        if self.phase == self.PHASE_GROW and action_idx in (self.POLICY_PRUNE_LAYER, self.VALUE_PRUNE_LAYER):
            print(f"[Architect] Deferred prune action {action_idx} during grow phase")
            return False

        if action_idx == self.POLICY_ADD_LAYER:
            g_pol = agent_graphs['policy']
            target_edge = next((e for e in g_pol.edges if e.to_id == "pol_out"), None)
            if target_edge:
                mutated = True
                self.mutation_count += 1
                new_lin_id = f"pol_mut_lin_{self.mutation_count}"
                new_relu_id = f"pol_mut_relu_{self.mutation_count}"
                g_pol.add_node(new_lin_id, "linear", {"in_features": 256, "out_features": 256})
                g_pol.add_node(new_relu_id, "relu")
                old_from = target_edge.from_id
                g_pol.edges.remove(target_edge)
                g_pol.add_edge(old_from, new_lin_id)
                g_pol.add_edge(new_lin_id, new_relu_id)
                g_pol.add_edge(new_relu_id, "pol_out")
                print(f"[Architect] Mutated Policy: Added layers {new_lin_id}, {new_relu_id}")

        elif action_idx == self.VALUE_ADD_LAYER:
            g_val = agent_graphs['value']
            target_edge = next((e for e in g_val.edges if e.to_id == "val_out"), None)
            if target_edge:
                mutated = True
                self.mutation_count += 1
                new_lin_id = f"val_mut_lin_{self.mutation_count}"
                new_relu_id = f"val_mut_relu_{self.mutation_count}"
                g_val.add_node(new_lin_id, "linear", {"in_features": 256, "out_features": 256})
                g_val.add_node(new_relu_id, "relu")
                old_from = target_edge.from_id
                g_val.edges.remove(target_edge)
                g_val.add_edge(old_from, new_lin_id)
                g_val.add_edge(new_lin_id, new_relu_id)
                g_val.add_edge(new_relu_id, "val_out")
                print(f"[Architect] Mutated Value: Added layers {new_lin_id}, {new_relu_id}")

        elif action_idx == self.POLICY_WIDEN:
            g_pol = agent_graphs['policy']
            target_node_id = "pol_lin1"
            if target_node_id in g_pol.nodes:
                node = g_pol.nodes[target_node_id]
                new_dim = node.config['out_features'] + 64
                node.config['out_features'] = new_dim
                next_node = self._find_next_layer_with_in_features(g_pol, target_node_id)
                if next_node:
                    next_node.config['in_features'] = new_dim
                mutated = True
                print(f"[Architect] Mutated Policy: Widened {target_node_id} to {new_dim}")

        elif action_idx == self.VALUE_WIDEN:
            g_val = agent_graphs['value']
            target_node_id = "val_lin1"
            if target_node_id in g_val.nodes:
                node = g_val.nodes[target_node_id]
                new_dim = node.config['out_features'] + 64
                node.config['out_features'] = new_dim
                next_node = self._find_next_layer_with_in_features(g_val, target_node_id)
                if next_node:
                    next_node.config['in_features'] = new_dim
                mutated = True
                print(f"[Architect] Mutated Value: Widened {target_node_id} to {new_dim}")

        elif action_idx == self.POLICY_PRUNE_LAYER:
            mutated = self._prune_hidden_layer(agent_graphs['policy'], "pol_lin1", "Policy Prune")

        elif action_idx == self.VALUE_PRUNE_LAYER:
            mutated = self._prune_hidden_layer(agent_graphs['value'], "val_lin1", "Value Prune")

        return mutated

    def transfer_weights(self, old_net, new_net, graph_def=None):
        old_layers = old_net.layers if hasattr(old_net, 'layers') else old_net.net.layers
        new_layers = new_net.layers if hasattr(new_net, 'layers') else new_net.net.layers

        edge_map = {e.to_id: e.from_id for e in graph_def.edges} if graph_def else {}
        kept_out_indices = {}

        def get_source_layer(node_id):
            curr = node_id
            while curr in edge_map:
                curr = edge_map[curr]
                if curr in old_layers and hasattr(old_layers[curr], 'weight'):
                    return curr
            return None

        for name, old_module in old_layers.items():
            if name not in new_layers:
                continue
            new_module = new_layers[name]

            if isinstance(old_module, nn.Linear) and isinstance(new_module, nn.Linear):
                out_old, in_old = old_module.weight.shape
                out_new, in_new = new_module.weight.shape

                if out_old == out_new and in_old == in_new:
                    new_module.weight.data.copy_(old_module.weight.data)
                    if old_module.bias is not None and new_module.bias is not None:
                        new_module.bias.data.copy_(old_module.bias.data)

                elif out_old >= out_new and in_old >= in_new:
                    in_indices = torch.arange(in_new)
                    if in_old > in_new and graph_def:
                        src_name = get_source_layer(name)
                        if src_name in kept_out_indices:
                            in_indices = kept_out_indices[src_name]

                    if out_old > out_new:
                        importances = get_channel_importances(old_module, norm='l1')
                        _, top_indices = torch.topk(importances, out_new)
                        out_indices = top_indices.sort().values
                        kept_out_indices[name] = out_indices
                    else:
                        out_indices = torch.arange(out_new)

                    new_module.weight.data.zero_()
                    new_module.weight.data.copy_(old_module.weight.data[out_indices][:, in_indices])
                    if old_module.bias is not None and new_module.bias is not None:
                        new_module.bias.data.copy_(old_module.bias.data[out_indices])
                else:
                    min_out = min(out_old, out_new)
                    min_in = min(in_old, in_new)
                    new_module.weight.data.zero_()
                    new_module.weight.data[:min_out, :min_in] = old_module.weight.data[:min_out, :min_in]
                    if old_module.bias is not None and new_module.bias is not None:
                        new_module.bias.data.zero_()
                        new_module.bias.data[:min_out] = old_module.bias.data[:min_out]

            elif isinstance(old_module, nn.Conv2d) and isinstance(new_module, nn.Conv2d):
                old_shape = old_module.weight.shape
                new_shape = new_module.weight.shape
                if old_shape == new_shape:
                    new_module.weight.data.copy_(old_module.weight.data)
                    if old_module.bias is not None and new_module.bias is not None:
                        new_module.bias.data.copy_(old_module.bias.data)
                else:
                    out_old, in_old, k_h, k_w = old_shape
                    out_new, in_new, _, _ = new_shape
                    if old_module.kernel_size == new_module.kernel_size and old_module.stride == new_module.stride and old_module.padding == new_module.padding:
                        in_indices = torch.arange(in_new)
                        if in_old > in_new and graph_def:
                            src_name = get_source_layer(name)
                            if src_name in kept_out_indices:
                                in_indices = kept_out_indices[src_name]

                        if out_old > out_new:
                            importances = get_channel_importances(old_module, norm='l1')
                            _, top_indices = torch.topk(importances, out_new)
                            out_indices = top_indices.sort().values
                            kept_out_indices[name] = out_indices
                        else:
                            out_indices = torch.arange(out_new)

                        new_module.weight.data.zero_()
                        new_module.weight.data.copy_(old_module.weight.data[out_indices][:, in_indices])
                        if old_module.bias is not None and new_module.bias is not None:
                            new_module.bias.data.copy_(old_module.bias.data[out_indices])

            else:
                try:
                    new_module.load_state_dict(old_module.state_dict())
                except Exception:
                    pass

        for name, new_module in new_layers.items():
            if name not in old_layers and isinstance(new_module, nn.Linear) and "mut_lin" in name:
                with torch.no_grad():
                    nn.init.zeros_(new_module.weight)
                    min_dim = min(new_module.weight.shape[0], new_module.weight.shape[1])
                    new_module.weight[:min_dim, :min_dim].copy_(torch.eye(min_dim))
                    if new_module.bias is not None:
                        nn.init.zeros_(new_module.bias)
