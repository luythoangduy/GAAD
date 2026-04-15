"""
Graph-Anchored Anomaly Detection — Graph Definitions & Text Anchors
=======================================================================
Offline Phase:
  - Each MVTec class has a set of Nodes (physical structures) and Edges
    (normal relational descriptions between nodes).
  - Edges are embedded via CLIP text encoder → stored as learnable Parameters
    that are initialized from those embeddings but updated during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import os


# ---------------------------------------------------------------------------
# 1.  Per-class graph descriptions
# ---------------------------------------------------------------------------

class ClassGraph:
    """Holds node names and edge descriptions for one anomaly-detection class."""

    def __init__(self, nodes: List[str], edges: List[str]):
        self.nodes = nodes          # fixed — NOT learned
        self.edges = edges          # text descriptions used to init anchor vectors


# MVTec AD — 15 classes
MVTEC_CLASS_GRAPHS: Dict[str, ClassGraph] = {
    "carpet": ClassGraph(
        nodes=["fiber", "weave_pattern", "surface", "color"],
        edges=[
            "a photo of carpet with uniform weave pattern",
            "a photo of carpet showing consistent fiber texture",
            "a photo of carpet with regular grid structure",
            "a photo of carpet surface with homogeneous color distribution",
            "a photo of carpet with smooth interlaced fiber arrangement",
            "a photo of normal carpet without any defects or damage",
        ],
    ),
    "grid": ClassGraph(
        nodes=["wire", "cell", "intersection", "surface"],
        edges=[
            "a photo of metal grid with regular cell spacing",
            "a photo of grid showing uniform wire thickness",
            "a photo of grid with clean intersection joints",
            "a photo of normal metal grid surface",
            "a photo of grid with consistent rectangular cell pattern",
            "a photo of grid wires without breaks or bends",
        ],
    ),
    "leather": ClassGraph(
        nodes=["surface", "texture", "grain", "color", "edge"],
        edges=[
            "a photo of leather with uniform grain texture",
            "a photo of leather surface with consistent color tone",
            "a photo of normal leather without scratches or cuts",
            "a photo of leather showing natural grain pattern",
            "a photo of leather with smooth homogeneous surface",
            "a photo of leather edge with clean straight boundary",
        ],
    ),
    "tile": ClassGraph(
        nodes=["surface", "glaze", "edge", "pattern", "color"],
        edges=[
            "a photo of tile with uniform glaze coating",
            "a photo of tile showing consistent surface pattern",
            "a photo of normal tile without cracks or chips",
            "a photo of tile with clean edges and corners",
            "a photo of tile surface with homogeneous color",
            "a photo of tile with smooth regular texture",
        ],
    ),
    "wood": ClassGraph(
        nodes=["grain", "surface", "knot", "color", "fiber"],
        edges=[
            "a photo of wood with natural grain pattern",
            "a photo of wood surface with consistent fiber direction",
            "a photo of normal wood without splits or cracks",
            "a photo of wood showing uniform color distribution",
            "a photo of wood with smooth planed surface",
            "a photo of wood grain without anomalous marks",
        ],
    ),
    "bottle": ClassGraph(
        nodes=["body", "neck", "surface", "label", "cap"],
        edges=[
            "a photo of bottle body with uniform smooth surface",
            "a photo of bottle neck with clean cylindrical shape",
            "a photo of bottle without scratches or contamination",
            "a photo of bottle with consistent transparent surface",
            "a photo of bottle showing normal round cross-section",
            "a photo of normal glass bottle without defects",
        ],
    ),
    "cable": ClassGraph(
        nodes=["wire", "insulation", "connector", "surface", "arrangement"],
        edges=[
            "a photo of cable with proper wire arrangement",
            "a photo of cable insulation without cuts or damage",
            "a photo of cable showing correct color coding",
            "a photo of cable wires with uniform parallel layout",
            "a photo of cable surface without abrasion",
            "a photo of normal cable without missing wires",
        ],
    ),
    "capsule": ClassGraph(
        nodes=["body", "cap", "surface", "joint", "color"],
        edges=[
            "a photo of capsule with smooth uniform surface",
            "a photo of capsule body without scratches or dents",
            "a photo of capsule showing consistent color",
            "a photo of capsule cap and body properly joined",
            "a photo of normal pharmaceutical capsule",
            "a photo of capsule with clean edge profile",
        ],
    ),
    "hazelnut": ClassGraph(
        nodes=["shell", "surface", "texture", "tip", "color"],
        edges=[
            "a photo of hazelnut with natural shell texture",
            "a photo of hazelnut surface without cracks",
            "a photo of hazelnut showing uniform brown color",
            "a photo of hazelnut with normal tip structure",
            "a photo of normal hazelnut without damage",
            "a photo of hazelnut shell with consistent roughness",
        ],
    ),
    "metal_nut": ClassGraph(
        nodes=["surface", "hole", "thread", "edge", "coating"],
        edges=[
            "a photo of metal nut with uniform surface finish",
            "a photo of metal nut hole with clean threading",
            "a photo of metal nut edge without burrs",
            "a photo of metal nut with consistent coating",
            "a photo of normal metal nut without corrosion",
            "a photo of metal nut with proper hexagonal shape",
        ],
    ),
    "pill": ClassGraph(
        nodes=["surface", "coating", "edge", "imprint", "color"],
        edges=[
            "a photo of pill with smooth coating surface",
            "a photo of pill showing uniform color",
            "a photo of pill edge without chips or cracks",
            "a photo of normal pharmaceutical pill",
            "a photo of pill with clean imprint marking",
            "a photo of pill with consistent circular shape",
        ],
    ),
    "screw": ClassGraph(
        nodes=["head", "thread", "tip", "shank", "surface"],
        edges=[
            "a photo of screw with properly formed threads",
            "a photo of screw head without damage",
            "a photo of screw tip with correct point geometry",
            "a photo of screw shank with uniform diameter",
            "a photo of normal metal screw without defects",
            "a photo of screw with clean surface finish",
        ],
    ),
    "toothbrush": ClassGraph(
        nodes=["bristles", "head", "handle", "tuft", "arrangement"],
        edges=[
            "a photo of toothbrush with uniform bristle arrangement",
            "a photo of toothbrush head with complete tufts",
            "a photo of toothbrush bristles at correct angle",
            "a photo of normal toothbrush without missing bristles",
            "a photo of toothbrush with consistent handle color",
            "a photo of toothbrush showing regular bristle density",
        ],
    ),
    "transistor": ClassGraph(
        nodes=["lead", "body", "surface", "marking", "base"],
        edges=[
            "a photo of transistor with straight leads properly spaced",
            "a photo of transistor body without cracks",
            "a photo of transistor surface with clear markings",
            "a photo of normal electronic transistor",
            "a photo of transistor base without damage",
            "a photo of transistor with correct lead geometry",
        ],
    ),
    "zipper": ClassGraph(
        nodes=["teeth", "slider", "tape", "tooth_spacing", "surface"],
        edges=[
            "a photo of zipper with evenly spaced teeth",
            "a photo of zipper teeth properly interlocked",
            "a photo of zipper tape without fraying",
            "a photo of normal zipper without broken teeth",
            "a photo of zipper with consistent tooth alignment",
            "a photo of zipper slider in correct position",
        ],
    ),
}


# ---------------------------------------------------------------------------
# 2.  Build learnable edge anchors from CLIP text embeddings
# ---------------------------------------------------------------------------

def build_text_anchors(
    class_graphs: Dict[str, ClassGraph],
    clip_model,
    clip_tokenizer,
    device: str = "cuda",
) -> nn.ParameterDict:
    """
    Encode every edge description with CLIP text encoder and return a
    ParameterDict mapping class_name → nn.Parameter[num_edges, D_clip].

    The parameters are learnable (requires_grad=True) so they can be
    updated during training, but are warm-started from CLIP embeddings.
    """
    import clip  # openai/clip

    edge_anchors = nn.ParameterDict()
    clip_model.eval()

    with torch.no_grad():
        for class_name, graph in class_graphs.items():
            tokens = clip_tokenizer(graph.edges).to(device)
            text_feats = clip_model.encode_text(tokens)          # [E, D]
            text_feats = F.normalize(text_feats.float(), dim=-1) # unit norm

            param = nn.Parameter(text_feats.clone(), requires_grad=True)
            # ParameterDict keys cannot contain '/', replace if needed
            safe_key = class_name.replace("/", "_")
            edge_anchors[safe_key] = param

    return edge_anchors


# ---------------------------------------------------------------------------
# 3.  Anchor cosine loss  (used during training)
# ---------------------------------------------------------------------------

def anchor_cosine_loss(
    adapter_out: torch.Tensor,              # [B, N_patches, D]
    anchors: torch.Tensor,                  # [E, D]
    hard_mining_p: float = 0.9,
    hard_mining_factor: float = 0.1,
) -> torch.Tensor:
    """
    For every patch, compute max cosine similarity to all edge anchors.
    Loss = mean(1 - max_sim).
    Hard mining: patches in the top-p% (easiest, highest sim) receive
    scaled-down gradients via register_hook.

    Args:
        adapter_out: adapter output vectors, L2-normalised.
        anchors:     edge anchor vectors, L2-normalised.
        hard_mining_p: fraction of easy patches to suppress (default 90%).
        hard_mining_factor: gradient scale for easy patches.
    """
    # Normalize both sides
    out_n = F.normalize(adapter_out, dim=-1)    # [B, N, D]
    anc_n = F.normalize(anchors, dim=-1)        # [E, D]

    # Max cosine similarity to any anchor — [B, N]
    # sim[b,n,e] = dot(out_n[b,n], anc_n[e])
    sim = torch.einsum("bnd,ed->bne", out_n, anc_n)   # [B, N, E]
    max_sim, _ = sim.max(dim=-1)                        # [B, N]

    loss = (1.0 - max_sim).mean()

    # Hard-mining: suppress gradients on easy (high-sim) patches
    if hard_mining_p > 0 and adapter_out.requires_grad:
        with torch.no_grad():
            dist = 1.0 - max_sim.detach()               # [B, N]
            k = max(1, int(dist.numel() * (1 - hard_mining_p)))
            thresh = torch.topk(dist.reshape(-1), k=k)[0][-1]
            easy_mask = (dist < thresh).unsqueeze(-1)    # [B, N, 1]

        def _hook(grad, mask=easy_mask, factor=hard_mining_factor):
            grad = grad.clone()
            grad[mask.expand_as(grad)] *= factor
            return grad

        adapter_out.register_hook(_hook)

    return loss


# ---------------------------------------------------------------------------
# 4.  Inference scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_image_score(
    adapter_out: torch.Tensor,   # [B, N_patches, D]
    anchors: torch.Tensor,       # [E, D]
    top_ratio: float = 0.01,
) -> torch.Tensor:
    """
    Returns image-level anomaly score [B] as the mean distance of the
    top-k% most anomalous patches.
    """
    out_n = F.normalize(adapter_out, dim=-1)            # [B, N, D]
    anc_n = F.normalize(anchors, dim=-1)                # [E, D]

    sim = torch.einsum("bnd,ed->bne", out_n, anc_n)    # [B, N, E]
    max_sim, _ = sim.max(dim=-1)                        # [B, N]
    dist = 1.0 - max_sim                                # [B, N]

    k = max(1, int(dist.shape[1] * top_ratio))
    top_dist = torch.topk(dist, k=k, dim=1)[0]         # [B, k]
    return top_dist.mean(dim=1)                         # [B]


# ---------------------------------------------------------------------------
# 5.  Graph visualization  (called once after offline phase)
# ---------------------------------------------------------------------------

def _short_label(text: str, max_words: int = 5) -> str:
    """Shorten an edge description to a compact label for graph display."""
    words = text.replace("a photo of ", "").split()
    label = " ".join(words[:max_words])
    if len(words) > max_words:
        label += "…"
    return label


def visualize_all_graphs(
    class_graphs: Dict,
    save_dir: str = "./graphs",
    figsize=(10, 7),
    node_color: str = "#4A90D9",
    edge_color: str = "#E07B39",
    font_size: int = 9,
    print_fn=print,
) -> None:
    """
    Draw one figure per class showing the physical-structure graph.

    Layout  :  Nodes = physical components (circular layout).
               Edges = normal-state relationships (labeled with short description).
    Output  :  `save_dir/<class_name>.png`

    Args:
        class_graphs : dict returned by MVTEC_CLASS_GRAPHS or a subset of it.
        save_dir     : folder to write PNG files into (created if needed).
        figsize      : matplotlib figure size per class.
        node_color   : hex color for node circles.
        edge_color   : hex color for edge lines / labels.
        font_size    : font size for all text elements.
        print_fn     : logging callable (e.g. logger.info).
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (safe for servers)
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print_fn(f"[visualize_all_graphs] Missing dependency: {e}. "
                 "Install networkx and matplotlib to enable visualization.")
        return

    os.makedirs(save_dir, exist_ok=True)
    print_fn(f"[visualize_all_graphs] Saving graph images to: {save_dir}/")

    for cls_name, graph in class_graphs.items():
        G = nx.DiGraph()

        # ---- Add nodes ----
        for node in graph.nodes:
            G.add_node(node)

        # ---- Add edges (cycle: each node → next, edges labeled) ----
        # We connect nodes in a cycle to form a ring skeleton, then annotate
        # with edge descriptions as edge labels.
        n_nodes = len(graph.nodes)
        n_edges = len(graph.edges)

        # Assign each edge description to a pair of adjacent nodes (modulo)
        edge_label_map = {}
        for idx, edge_desc in enumerate(graph.edges):
            src = graph.nodes[idx % n_nodes]
            dst = graph.nodes[(idx + 1) % n_nodes]
            short = _short_label(edge_desc)
            # allow parallel edges in DiGraph by adding suffix key
            key = (src, dst)
            if key in edge_label_map:
                # place on next available pair
                src = graph.nodes[(idx + 1) % n_nodes]
                dst = graph.nodes[(idx + 2) % n_nodes]
                key = (src, dst)
            G.add_edge(src, dst)
            edge_label_map[key] = short

        # ---- Layout ----
        pos = nx.circular_layout(G)

        # ---- Draw ----
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#1A1A2E")
        fig.patch.set_facecolor("#1A1A2E")

        # Edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_color,
            arrows=True,
            arrowsize=18,
            width=1.8,
            connectionstyle="arc3,rad=0.12",
        )
        # Edge labels
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_label_map, ax=ax,
            font_size=font_size - 1,
            font_color="#F5A623",
            bbox=dict(boxstyle="round,pad=0.2", fc="#1A1A2E", ec="none", alpha=0.7),
            rotate=False,
        )
        # Nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_color,
            node_size=1800,
            alpha=0.95,
        )
        # Node labels
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=font_size,
            font_color="white",
            font_weight="bold",
        )

        # ---- Title & legend ----
        ax.set_title(
            f"Normal-State Graph — {cls_name.replace('_', ' ').title()}",
            fontsize=13, fontweight="bold", color="white", pad=14,
        )
        node_patch = mpatches.Patch(color=node_color, label="Node (physical structure)")
        edge_patch = mpatches.Patch(color=edge_color, label="Edge (normal relationship anchor)")
        ax.legend(
            handles=[node_patch, edge_patch],
            loc="lower right", fontsize=font_size - 1,
            facecolor="#2C2C54", edgecolor="none", labelcolor="white",
        )
        ax.axis("off")

        out_path = os.path.join(save_dir, f"{cls_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print_fn(f"  [graph] {cls_name} -> {out_path}")

    print_fn(f"[visualize_all_graphs] Done — {len(class_graphs)} graphs saved.")
