"""
GAAD — Graph-Anchored Anomaly Detection
Training script for MVTec AD (unified, image-level metrics only)
================================================================
Replaces knowledge distillation (dinomaly_mvtec_uni.py) with the
graph-anchored adapter flow:

  Offline  : CLIP text encoder → learnable edge_anchors per class
  Training : DINO (frozen, multi-layer) → adapter blocks →
             projection → anchor cosine loss
  Inference: max cosine distance to anchors → image score
             (only I-AUROC, I-AP, I-F1 are reported)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import logging
import argparse
import warnings
from functools import partial
from torch.utils.data import DataLoader, ConcatDataset

# -- Project imports --
from dataset import get_data_transforms, MVTecDataset
from models import vit_encoder
from models.uad import GraphAnomalyAdapter
from models.graph_anchors import (
    MVTEC_CLASS_GRAPHS,
    build_text_anchors,
    anchor_cosine_loss,
    visualize_all_graphs,
)
from models.vision_transformer import Block as VitBlock, LinearAttention2
from dinov1.utils import trunc_normal_
from optimizers import StableAdamW
from utils import WarmCosineScheduler, evaluation_image_only

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_logger(name, save_path=None, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    fmt = logging.Formatter("%(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_path, "log.txt"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Build learnable edge anchors (warm-started from CLIP)
# ---------------------------------------------------------------------------

def build_edge_anchors(item_list, device):
    """
    Load CLIP ViT-B/16, encode edge descriptions for each class, and return
    a ParameterDict of learnable anchors.
    """
    try:
        import clip
        clip_model, _ = clip.load("ViT-B/16", device=device)
        tokenizer = clip.tokenize
        print_fn("CLIP loaded successfully → building text anchors …")
        anchor_dict = build_text_anchors(
            {k: MVTEC_CLASS_GRAPHS[k] for k in item_list},
            clip_model,
            tokenizer,
            device=device,
        )
        clip_dim = 512  # ViT-B/16
        del clip_model
        torch.cuda.empty_cache()
    except Exception as e:
        print_fn(f"CLIP not available ({e}). Falling back to random anchors (dim=512).")
        anchor_dict = nn.ParameterDict()
        clip_dim = 512
        for cls_name in item_list:
            num_edges = len(MVTEC_CLASS_GRAPHS[cls_name].edges)
            rand_anc = F.normalize(
                torch.randn(num_edges, clip_dim, device=device), dim=-1
            )
            anchor_dict[cls_name.replace("/", "_")] = nn.Parameter(rand_anc)

    return anchor_dict, clip_dim


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(item_list):
    setup_seed(1)

    total_iters = 10_000
    batch_size  = 16
    image_size  = 448
    crop_size   = 392

    # ------------------------------------------------------------------ data
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list  = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, "train")
        test_path  = os.path.join(args.data_path, item)

        td = ImageFolder(root=train_path, transform=data_transform)
        td.class_to_idx = {item: i}
        td.samples = [(s[0], i) for s in td.samples]

        test_data = MVTecDataset(
            root=test_path, transform=data_transform,
            gt_transform=gt_transform, phase="test",
        )
        train_data_list.append(td)
        test_data_list.append(test_data)

    train_data       = ConcatDataset(train_data_list)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True,
    )
    print_fn(f"Train images: {len(train_data)}")

    # ------------------------------------------------------- DINO backbone
    encoder_name = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]   # 8 layers → multi-level feats

    encoder = vit_encoder.load(encoder_name)
    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("Unsupported encoder size.")

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    # ------------------------------------------------------- Anchor anchors (offline)
    edge_anchors, clip_dim = build_edge_anchors(item_list, device)
    print_fn(f"Anchor space dim: {clip_dim}")

    # ------------------------------------------------------- Visualize graphs
    if not args.no_viz:
        graph_save_dir = os.path.join(args.save_dir, args.save_name, "graphs")
        visualize_all_graphs(
            {k: MVTEC_CLASS_GRAPHS[k] for k in item_list},
            save_dir=graph_save_dir,
            print_fn=print_fn,
        )

    # ------------------------------------------------------- Adapter blocks
    num_adapter_blocks = 4   # lightweight — fewer than original 8-block decoder
    adapter_blocks = nn.ModuleList([
        VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        for _ in range(num_adapter_blocks)
    ])

    # Projection: DINO embed_dim → CLIP anchor dim
    proj = nn.Linear(embed_dim, clip_dim, bias=False)

    # ------------------------------------------------------- Full model
    model = GraphAnomalyAdapter(
        encoder=encoder,
        adapter=adapter_blocks,
        proj=proj,
        target_layers=target_layers,
    ).to(device)

    # Trainable components: adapter + proj + edge_anchors
    trainable = nn.ModuleList([adapter_blocks, nn.ModuleList([proj])])

    # Weight init for adapter & proj
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    print_fn(f"Trainable adapter+proj params: {count_parameters(trainable):,}")

    # edge_anchors are also trainable
    anchor_params = list(edge_anchors.parameters())
    print_fn(f"Anchor parameters total: {sum(p.numel() for p in anchor_params):,}")

    optimizer = StableAdamW(
        [
            {"params": trainable.parameters(), "lr": 2e-3},
            {"params": anchor_params,          "lr": 1e-4},   # slower for anchors
        ],
        betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10,
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=2e-3, final_value=2e-4,
        total_iters=total_iters, warmup_iters=100,
    )

    # ------------------------------------------------------- Training loop
    it  = 0
    p_final = 0.9   # hard mining ramp target

    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []

        for img, label in train_dataloader:
            img   = img.to(device)
            label = label.to(device)

            adapter_out = model(img)   # [B, N, D_anchor]

            # --- Gather anchors for each sample in batch ---
            # All samples in a batch may come from different classes;
            # compute per-sample loss and average.
            loss = torch.tensor(0.0, device=device)
            for b in range(img.shape[0]):
                cls_name = item_list[label[b].item()].replace("/", "_")
                anchors  = edge_anchors[cls_name]   # [E, D]
                p_curr   = min(p_final * it / 1000, p_final)
                loss += anchor_cosine_loss(
                    adapter_out[b].unsqueeze(0),    # [1, N, D]
                    anchors,
                    hard_mining_p=p_curr,
                    hard_mining_factor=0.1,
                )
            loss = loss / img.shape[0]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(trainable.parameters()) + anchor_params, max_norm=0.1
            )
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(loss.item())

            # ---- Periodic evaluation (image metrics only) ----
            if (it + 1) % 5000 == 0:
                auroc_list, ap_list, f1_list = [], [], []

                for item_name, test_data in zip(item_list, test_data_list):
                    test_dl = DataLoader(
                        test_data, batch_size=batch_size,
                        shuffle=False, num_workers=4,
                    )
                    metrics = evaluation_image_only(
                        model, test_dl, device,
                        edge_anchors_dict=edge_anchors,
                        class_list=item_list,
                        top_ratio=0.01,
                    )
                    auroc_list.append(metrics["auroc"])
                    ap_list.append(metrics["ap"])
                    f1_list.append(metrics["f1"])
                    print_fn(
                        f"{item_name}: I-AUROC:{metrics['auroc']:.4f} "
                        f"I-AP:{metrics['ap']:.4f} I-F1:{metrics['f1']:.4f}"
                    )

                print_fn(
                    f"Mean : I-AUROC:{np.mean(auroc_list):.4f} "
                    f"I-AP:{np.mean(ap_list):.4f} "
                    f"I-F1:{np.mean(f1_list):.4f}"
                )
                model.train()

            it += 1
            if it == total_iters:
                break

        print_fn(
            f"Epoch end | iter [{it}/{total_iters}] "
            f"loss: {np.mean(loss_list):.4f}"
        )

    # ---- Final evaluation ----
    print_fn("\n=== Final Evaluation ===")
    auroc_list, ap_list, f1_list = [], [], []
    for item_name, test_data in zip(item_list, test_data_list):
        test_dl = DataLoader(
            test_data, batch_size=batch_size,
            shuffle=False, num_workers=4,
        )
        metrics = evaluation_image_only(
            model, test_dl, device,
            edge_anchors_dict=edge_anchors,
            class_list=item_list,
            top_ratio=0.01,
        )
        auroc_list.append(metrics["auroc"])
        ap_list.append(metrics["ap"])
        f1_list.append(metrics["f1"])
        print_fn(
            f"{item_name}: I-AUROC:{metrics['auroc']:.4f} "
            f"I-AP:{metrics['ap']:.4f} I-F1:{metrics['f1']:.4f}"
        )

    print_fn(
        f"\nMean: I-AUROC:{np.mean(auroc_list):.4f} "
        f"I-AP:{np.mean(ap_list):.4f} "
        f"I-F1:{np.mean(f1_list):.4f}"
    )

    # Save checkpoint
    save_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "edge_anchors": {k: v.data for k, v in edge_anchors.items()},
            "item_list": item_list,
        },
        os.path.join(save_dir, "model.pth"),
    )
    print_fn(f"Checkpoint saved to {save_dir}/model.pth")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    parser = argparse.ArgumentParser(description="GAAD — Graph-Anchored AD on MVTec")
    parser.add_argument("--data_path",  type=str, default="../mvtec_anomaly_detection")
    parser.add_argument("--save_dir",   type=str, default="./saved_results")
    parser.add_argument("--save_name",  type=str,
                        default="gaad_mvtec_uni_dinov2br_c392_en29_ada4_clip512_it10k")
    parser.add_argument("--gpu",        type=str, default="cuda:0")
    parser.add_argument("--no_viz",     action="store_true",
                        help="Skip graph visualization after offline phase.")
    args = parser.parse_args()

    item_list = [
        "carpet", "grid", "leather", "tile", "wood",
        "bottle", "cable", "capsule", "hazelnut", "metal_nut",
        "pill", "screw", "toothbrush", "transistor", "zipper",
    ]

    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device   = args.gpu if torch.cuda.is_available() else "cpu"
    print_fn(f"Device: {device}")

    train(item_list)
