import torch
import torch.nn as nn
from modules.graph import GraphDef, ModuleBuilder, get_channel_importances

def test():
    # Build a simple CNN + MLP graph
    graph = GraphDef()
    
    # input node is implicit: "input"
    # Node definitions
    graph.add_node("conv1", "conv2d", {"in_channels": 3, "out_channels": 16, "kernel_size": 8, "stride": 4})
    graph.add_node("relu1", "relu")
    graph.add_node("conv2", "conv2d", {"in_channels": 16, "out_channels": 32, "kernel_size": 4, "stride": 2})
    graph.add_node("relu2", "relu")
    graph.add_node("flatten", "flatten")
    graph.add_node("linear1", "linear", {"in_features": 32 * 6 * 6, "out_features": 256})
    graph.add_node("relu3", "relu")
    
    # Edges define the flow
    graph.add_edge("input", "conv1")
    graph.add_edge("conv1", "relu1")
    graph.add_edge("relu1", "conv2")
    graph.add_edge("conv2", "relu2")
    graph.add_edge("relu2", "flatten")
    graph.add_edge("flatten", "linear1")
    graph.add_edge("linear1", "relu3")
    
    model = ModuleBuilder.build(graph)
    print(model)
    
    # Test forward pass with dummy MiniGrid RGB observation shape (B, C, H, W)
    dummy_input = torch.randn(1, 3, 64, 64)
    out = model(dummy_input)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 256)
    print("Graph module build and forward pass: SUCCESS")

def test_mutation_identity():
    import copy
    from configurator.graph_modifier import GraphModifier
    
    # 1. Base graph
    g_pol = GraphDef()
    g_pol.add_node("pol_lin1", "linear", {"in_features": 128, "out_features": 256}).add_node("pol_relu1", "relu").add_node("pol_out", "linear", {"in_features": 256, "out_features": 7})
    g_pol.add_edge("input", "pol_lin1").add_edge("pol_lin1", "pol_relu1").add_edge("pol_relu1", "pol_out")
    
    old_model = ModuleBuilder.build(g_pol)
    dummy_input = torch.randn(2, 128)
    
    with torch.no_grad():
        out_old = old_model(dummy_input)
        
    # 2. Mutate (Widen Policy)
    modifier = GraphModifier()
    graphs = {'policy': g_pol}
    
    # action 3 = POLICY_WIDEN
    modifier.apply_action(graphs, 3)
    
    # 3. Build new model and transfer weights
    new_model = ModuleBuilder.build(g_pol)
    modifier.transfer_weights(old_model, new_model)
    
    # 4. Compare outputs
    with torch.no_grad():
        out_new = new_model(dummy_input)
        
    assert not torch.isnan(out_new).any(), "NaN found in output after mutation!"
    assert out_new.shape == out_old.shape, f"Output shape changed: {out_new.shape} vs {out_old.shape}"
    
    # Due to strict zero-padding, outputs should be exactly identical
    assert torch.allclose(out_old, out_new, atol=1e-6), "Outputs are not identical after widening mutation!"
    print("Graph mutation identity preservation: SUCCESS")


def test_pruning_preserves_shape():
    from configurator.graph_modifier import GraphModifier

    g_pol = GraphDef()
    g_pol.add_node("pol_lin1", "linear", {"in_features": 128, "out_features": 256}).add_node("pol_relu1", "relu").add_node("pol_out", "linear", {"in_features": 256, "out_features": 7})
    g_pol.add_edge("input", "pol_lin1").add_edge("pol_lin1", "pol_relu1").add_edge("pol_relu1", "pol_out")

    old_model = ModuleBuilder.build(g_pol)
    dummy_input = torch.randn(2, 128)
    with torch.no_grad():
        out_old = old_model(dummy_input)

    modifier = GraphModifier()
    graphs = {'policy': g_pol}
    modifier.apply_action(graphs, 5)

    new_model = ModuleBuilder.build(g_pol)
    modifier.transfer_weights(old_model, new_model, graphs['policy'])

    with torch.no_grad():
        out_new = new_model(dummy_input)

    assert out_new.shape == out_old.shape, f"Output shape changed after pruning: {out_new.shape} vs {out_old.shape}"
    assert not torch.isnan(out_new).any(), "NaN found in output after pruning mutation!"
    print("Graph pruning mutation shape and stability: SUCCESS")


def test_meta_phase_scheduler():
    from configurator.graph_modifier import GraphModifier

    modifier = GraphModifier()
    assert modifier.set_meta_phase(0, cycle_length=4, prune_phase_length=2) == modifier.PHASE_PRUNE
    assert modifier.set_meta_phase(1, cycle_length=4, prune_phase_length=2) == modifier.PHASE_PRUNE
    assert modifier.set_meta_phase(2, cycle_length=4, prune_phase_length=2) == modifier.PHASE_GROW
    assert modifier.set_meta_phase(3, cycle_length=4, prune_phase_length=2) == modifier.PHASE_GROW

    g_pol = GraphDef()
    g_pol.add_node("pol_lin1", "linear", {"in_features": 128, "out_features": 256}).add_node("pol_relu1", "relu").add_node("pol_out", "linear", {"in_features": 256, "out_features": 7})
    g_pol.add_edge("input", "pol_lin1").add_edge("pol_lin1", "pol_relu1").add_edge("pol_relu1", "pol_out")

    modifier.set_meta_phase(0, cycle_length=4, prune_phase_length=2)
    assert modifier.apply_action({'policy': g_pol, 'value': g_pol}, GraphModifier.POLICY_WIDEN) is False
    assert modifier.apply_action({'policy': g_pol, 'value': g_pol}, GraphModifier.POLICY_PRUNE_LAYER) is True

    modifier.set_meta_phase(2, cycle_length=4, prune_phase_length=2)
    assert modifier.apply_action({'policy': g_pol, 'value': g_pol}, GraphModifier.POLICY_PRUNE_LAYER) is False
    print("Meta-phase scheduler behavior: SUCCESS")


def test_channel_importances():
    linear = nn.Linear(10, 8)
    importances = get_channel_importances(linear, norm='l1')
    assert importances.shape == (8,)
    assert (importances >= 0).all()

    conv = nn.Conv2d(3, 5, kernel_size=3)
    importances = get_channel_importances(conv, norm='l2')
    assert importances.shape == (5,)
    assert (importances >= 0).all()
    print("Channel importance utility: SUCCESS")


def test_skip_connection():
    from configurator.graph_modifier import GraphModifier

    # Build a policy graph: input(128) -> lin(128->256) -> relu -> out(256->7)
    g_pol = GraphDef()
    g_pol.add_node("pol_lin1", "linear", {"in_features": 128, "out_features": 256})
    g_pol.add_node("pol_relu1", "relu")
    g_pol.add_node("pol_out", "linear", {"in_features": 256, "out_features": 7})
    g_pol.add_edge("input", "pol_lin1")
    g_pol.add_edge("pol_lin1", "pol_relu1")
    g_pol.add_edge("pol_relu1", "pol_out")

    old_model = ModuleBuilder.build(g_pol)
    dummy_input = torch.randn(2, 128)
    with torch.no_grad():
        out_old = old_model(dummy_input)

    # Apply skip connection: input -> proj(128->256) -> add + main_path -> pol_out
    modifier = GraphModifier()
    graphs = {'policy': g_pol}
    mutated = modifier.apply_action(graphs, GraphModifier.POLICY_ADD_SKIP)
    assert mutated, "Skip connection mutation should have been applied"

    # Verify graph structure: should have add node and projection
    add_nodes = [n for n in g_pol.nodes.values() if n.layer_type == "add"]
    assert len(add_nodes) == 1, f"Expected 1 add node, got {len(add_nodes)}"

    proj_nodes = [n for n in g_pol.nodes.values() if n.id.startswith("skip_proj")]
    assert len(proj_nodes) == 1, f"Expected 1 projection node (128 != 256), got {len(proj_nodes)}"
    assert proj_nodes[0].config['in_features'] == 128
    assert proj_nodes[0].config['out_features'] == 256

    # Build and test forward pass
    new_model = ModuleBuilder.build(g_pol)
    modifier.transfer_weights(old_model, new_model, g_pol)

    with torch.no_grad():
        out_new = new_model(dummy_input)

    assert out_new.shape == out_old.shape, f"Shape mismatch: {out_new.shape} vs {out_old.shape}"
    assert not torch.isnan(out_new).any(), "NaN found after skip mutation"
    print("Skip connection mutation: SUCCESS")


def test_skip_duplicate_prevention():
    from configurator.graph_modifier import GraphModifier

    g_pol = GraphDef()
    g_pol.add_node("pol_lin1", "linear", {"in_features": 128, "out_features": 256})
    g_pol.add_node("pol_relu1", "relu")
    g_pol.add_node("pol_out", "linear", {"in_features": 256, "out_features": 7})
    g_pol.add_edge("input", "pol_lin1")
    g_pol.add_edge("pol_lin1", "pol_relu1")
    g_pol.add_edge("pol_relu1", "pol_out")

    modifier = GraphModifier()
    graphs = {'policy': g_pol}

    # First skip should succeed
    assert modifier.apply_action(graphs, GraphModifier.POLICY_ADD_SKIP) is True
    # Second skip should be blocked (add node already feeds into pol_out)
    assert modifier.apply_action(graphs, GraphModifier.POLICY_ADD_SKIP) is False
    print("Skip duplicate prevention: SUCCESS")


def test_add_module_direct():
    """Test AddModule directly with a hand-built multi-input graph."""
    from modules.graph import AddModule

    g = GraphDef()
    g.add_node("lin_a", "linear", {"in_features": 8, "out_features": 16})
    g.add_node("lin_b", "linear", {"in_features": 8, "out_features": 16})
    g.add_node("add1", "add", {})
    g.add_node("out", "linear", {"in_features": 16, "out_features": 4})
    g.add_edge("input", "lin_a")
    g.add_edge("input", "lin_b")
    g.add_edge("lin_a", "add1")
    g.add_edge("lin_b", "add1")
    g.add_edge("add1", "out")

    model = ModuleBuilder.build(g)
    dummy = torch.randn(3, 8)
    out = model(dummy)
    assert out.shape == (3, 4), f"Expected (3, 4), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in AddModule output"
    print("AddModule direct multi-input DAG: SUCCESS")


if __name__ == "__main__":
    test()
    test_mutation_identity()
    test_pruning_preserves_shape()
    test_meta_phase_scheduler()
    test_channel_importances()
    test_skip_connection()
    test_skip_duplicate_prevention()
    test_add_module_direct()

