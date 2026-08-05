from __future__ import annotations

import torch


def coalesce_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    linear = edge_index[0] * num_nodes + edge_index[1]
    unique = torch.unique(linear)
    return torch.stack([unique // num_nodes, unique % num_nodes], dim=0)


def build_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    add_self_loops: bool = True,
) -> torch.Tensor:
    edge_index = edge_index.to(device)
    if add_self_loops:
        loops = torch.arange(num_nodes, device=device)
        loop_index = torch.stack([loops, loops], dim=0)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_index = coalesce_edge_index(edge_index, num_nodes)
    row, col = edge_index
    degree = torch.bincount(row, minlength=num_nodes).to(torch.float32)
    inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
    values = inv_sqrt[row] * inv_sqrt[col]
    adjacency = torch.sparse_coo_tensor(
        edge_index,
        values,
        size=(num_nodes, num_nodes),
        device=device,
        check_invariants=True,
    )
    return adjacency.coalesce()


def build_normalized_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    edge_index = coalesce_edge_index(edge_index.to(device), num_nodes)
    row, col = edge_index
    non_self = row != col
    row = row[non_self]
    col = col[non_self]
    degree = torch.bincount(row, minlength=num_nodes).to(torch.float32)
    inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
    off_values = -(inv_sqrt[row] * inv_sqrt[col])
    diag = torch.arange(num_nodes, device=device)
    indices = torch.cat([torch.stack([row, col]), torch.stack([diag, diag])], dim=1)
    values = torch.cat([off_values, torch.ones(num_nodes, device=device)])
    laplacian = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        device=device,
        check_invariants=True,
    )
    return laplacian.coalesce()


def node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = coalesce_edge_index(edge_index.cpu(), num_nodes)
    row, col = edge_index
    degree = torch.bincount(row[row != col], minlength=num_nodes)
    return degree
