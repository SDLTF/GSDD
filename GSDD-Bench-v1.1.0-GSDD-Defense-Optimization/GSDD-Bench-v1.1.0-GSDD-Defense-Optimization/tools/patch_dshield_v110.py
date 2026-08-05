from __future__ import annotations

import argparse
from pathlib import Path


MAIN_MARKER = "GSDD_BENCH_WEIGHTED_TRAINING_V110"
GCN_MARKER = "GSDD_BENCH_NODE_WEIGHTS_V110"


def patch_main(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MAIN_MARKER in text:
        return
    original = text

    cli_anchor = "    parser.add_argument('--gsdd_train_idx_override', type=str, default='')\n"
    if cli_anchor not in text:
        raise RuntimeError("main.py GSDD CLI anchor not found; apply the v1.0 compatibility patch first")
    text = text.replace(
        cli_anchor,
        cli_anchor
        + "    parser.add_argument('--gsdd_train_weight_override', type=str, default='')  # "
        + MAIN_MARKER
        + "\n",
        1,
    )

    load_anchor = """    if args.gsdd_train_idx_override:
        train_node_idx = torch.load(args.gsdd_train_idx_override, map_location=device, weights_only=True).long().to(device)
        logger.info('[GSDD-Bench] using train-index override with %d nodes', int(train_node_idx.numel()))

"""
    if load_anchor not in text:
        raise RuntimeError("main.py train-index override anchor not found")
    weight_block = load_anchor + """    _gsdd_node_weights = None
    if args.gsdd_train_weight_override:
        _gsdd_node_weights = torch.load(
            args.gsdd_train_weight_override, map_location=device, weights_only=True
        ).float().reshape(-1).to(device)
        if int(_gsdd_node_weights.numel()) != int(feat.shape[0]):
            raise RuntimeError(
                'GSDD node-weight override length mismatch: '
                f'{int(_gsdd_node_weights.numel())} weights for {int(feat.shape[0])} nodes'
            )
        if not torch.isfinite(_gsdd_node_weights).all():
            raise RuntimeError('GSDD node weights contain NaN or infinity')
        if bool((_gsdd_node_weights <= 0).any()):
            raise RuntimeError('GSDD node weights must be strictly positive')
        logger.info(
            '[GSDD-Bench] using node-weight override: min=%.6f mean=%.6f max=%.6f',
            float(_gsdd_node_weights.min().item()),
            float(_gsdd_node_weights.mean().item()),
            float(_gsdd_node_weights.max().item()),
        )

"""
    text = text.replace(load_anchor, weight_block, 1)

    fit_anchor = """        benign_model.fit(feat, train_edge_index, train_edge_weight,
                         labels, train_node_idx, val_idx, train_iters=args.benign_epochs, verbose=True)
"""
    if fit_anchor not in text:
        raise RuntimeError("main.py benign GCN fit anchor not found")
    weighted_fit = """        if _gsdd_node_weights is None:
            benign_model.fit(feat, train_edge_index, train_edge_weight,
                             labels, train_node_idx, val_idx, train_iters=args.benign_epochs, verbose=True)
        else:
            if args.model != 'GCN':
                raise RuntimeError('GSDD soft weighting v1.1.0 currently supports model=GCN only')
            benign_model.fit(feat, train_edge_index, train_edge_weight,
                             labels, train_node_idx, val_idx, train_iters=args.benign_epochs,
                             verbose=True, node_weights=_gsdd_node_weights)
"""
    text = text.replace(fit_anchor, weighted_fit, 1)

    label_anchor = "args.defense_method if not args.gsdd_train_idx_override else 'GSDD'"
    if label_anchor in text:
        text = text.replace(
            label_anchor,
            "args.defense_method if not (args.gsdd_train_idx_override or args.gsdd_train_weight_override) else 'GSDD'",
            1,
        )

    path.with_suffix(".py.gsddbench.v110.bak").write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_gcn(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if GCN_MARKER in text:
        return
    original = text

    signature = "    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):\n"
    if signature not in text:
        raise RuntimeError("GCN.py fit signature anchor not found")
    text = text.replace(
        signature,
        "    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, node_weights=None):  # "
        + GCN_MARKER
        + "\n",
        1,
    )

    old_calls = """        if idx_val is None:
            self._train_without_val(features, edge_index, edge_weight, labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(features, edge_index, edge_weight, labels, idx_train, idx_val, train_iters, verbose)
"""
    new_calls = """        if node_weights is not None:
            node_weights = node_weights.to(self.device).float().reshape(-1)
            if int(node_weights.numel()) != int(features.shape[0]):
                raise ValueError('node_weights must contain one value per graph node')
        if idx_val is None:
            self._train_without_val(features, edge_index, edge_weight, labels, idx_train, train_iters, verbose, node_weights)
        else:
            self._train_with_val(features, edge_index, edge_weight, labels, idx_train, idx_val, train_iters, verbose, node_weights)
"""
    if old_calls not in text:
        raise RuntimeError("GCN.py fit dispatch anchor not found")
    text = text.replace(old_calls, new_calls, 1)

    sig_without = "    def _train_without_val(self, features, edge_index, edge_weight, labels, idx_train, train_iters, verbose):\n"
    sig_with = "    def _train_with_val(self, features, edge_index, edge_weight, labels, idx_train, idx_val, train_iters, verbose):\n"
    if sig_without not in text or sig_with not in text:
        raise RuntimeError("GCN.py internal training signature anchor not found")
    text = text.replace(
        sig_without,
        "    def _train_without_val(self, features, edge_index, edge_weight, labels, idx_train, train_iters, verbose, node_weights=None):\n",
        1,
    )
    text = text.replace(
        sig_with,
        "    def _train_with_val(self, features, edge_index, edge_weight, labels, idx_train, idx_val, train_iters, verbose, node_weights=None):\n",
        1,
    )

    loss_line = "            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])\n"
    if text.count(loss_line) != 2:
        raise RuntimeError(f"GCN.py expected two training loss anchors, found {text.count(loss_line)}")
    weighted_loss = """            per_node_loss = F.cross_entropy(output[idx_train], labels[idx_train], reduction='none')
            if node_weights is None:
                loss_train = per_node_loss.mean()
            else:
                selected_weights = node_weights[idx_train].clamp_min(1e-8)
                loss_train = (per_node_loss * selected_weights).sum() / selected_weights.sum()
"""
    text = text.replace(loss_line, weighted_loss, 2)

    path.with_suffix(".py.gsddbench.v110.bak").write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    node_root = repo / "NodeClassificationTasks"
    main_path = node_root / "main.py"
    gcn_path = node_root / "models" / "GCN.py"
    if not main_path.exists() or not gcn_path.exists():
        raise SystemExit(f"Invalid DShield checkout: {repo}")
    patch_main(main_path)
    patch_gcn(gcn_path)
    print(f"Patched DShield for GSDD v1.1.0 weighted training: {repo}")


if __name__ == "__main__":
    main()
