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

if __name__ == "__main__":
    test()
    test_mutation_identity()
    test_pruning_preserves_shape()
    test_meta_phase_scheduler()
    test_channel_importances()
