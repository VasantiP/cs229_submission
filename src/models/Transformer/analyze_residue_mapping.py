"""
analyze_residue_mapping.py — Map model attention weights back to PDB residues.

For each test sample, extracts residue attention weights (α) from a trained
ResidueVarianceClassifier or ResAttnTransformer, maps them back to PDB residue
numbers, annotates each residue with its structural role (helix / strand /
loop / binding-site), and renders images of the 3D structure colored by
attention weight — no PyMOL required.

Outputs (saved to <run_dir>/residue_mapping/):
  attention_by_residue.csv        — per-residue table with structural annotations
  aggregate_delta_alpha.png       — pooled Δα across all test samples
  per_receptor/<pdb>/
    01_attention_profile.png      — linear bar chart: mean α per class + Δα,
                                     colored by structural role
    02_structure_3views.png       — 3D Cα scatter (XY / XZ / YZ projections)
                                     colored by attention weight, top-k labelled
    03_binding_site_zoom.png      — attention at binding-site residues only

Structural role annotation (from PDB records, no external tools needed)
------------------------------------------------------------------------
  helix        — HELIX records in PDB header
  strand       — SHEET records in PDB header
  binding_site — protein residues within 4 Å of any HETATM ligand heavy atom
  loop/coil    — everything else (turns, loops, disordered regions)

Usage
-----
  python models/analyze_residue_mapping.py \\
      --checkpoint /home/jupyter/runs_resvar/<run>/checkpoint_best.pt \\
      --test-csv   /tmp/emb_bootstrap_test_chunks_newsplit.csv \\
      --meta-csv   data/processed/protein_level_test.csv \\
      --model-type resvar

  python models/analyze_residue_mapping.py \\
      --checkpoint /home/jupyter/runs_transformer/<run>/checkpoint_best.pt \\
      --test-csv   /tmp/emb_bootstrap_test_chunks_newsplit.csv \\
      --meta-csv   data/processed/protein_level_test.csv \\
      --model-type transformer
"""

import argparse
import re
import sys
import time
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.stats import fisher_exact

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))



def parse_pdb_id_chain(receptor_str):
    """'Receptor_name~3QAK_A' -> ('3QAK', 'A')"""
    parts = receptor_str.split("~")
    if len(parts) < 2:
        return None, None
    m = re.match(r"([A-Za-z0-9]{4})_([A-Za-z0-9]+)", parts[-1])
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).upper()


def download_pdb(pdb_id, pdb_dir):
    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    local = pdb_dir / f"{pdb_id.lower()}.pdb"
    if local.exists():
        return local
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                local.write_bytes(r.content)
                return local
        except requests.RequestException:
            pass
        time.sleep(2)
    return None



def parse_pdb(pdb_path, chain):
    """
    Parse a PDB file and return:
      residues   : list of dicts ordered as ESM-IF sees them
      helix_set  : set of (resnum, icode) in helix
      strand_set : set of (resnum, icode) in strand
      binding_set: set of (resnum, icode) within 4Å of any HETATM ligand
    """
    lines = Path(pdb_path).read_text().splitlines()

    helix_set      = set()
    strand_set     = set()
    active_site_set = set()   # from SITE records

    for line in lines:
        if line.startswith("HELIX "):
            # HELIX    1   1 GLY A    1  LEU A   32  ...
            h_chain = line[19].strip()
            if h_chain != chain:
                continue
            try:
                start = int(line[21:25].strip())
                end   = int(line[33:37].strip())
                for rn in range(start, end + 1):
                    helix_set.add((rn, ""))
            except ValueError:
                pass

        elif line.startswith("SHEET "):
            # SHEET    1   A 5 VAL A  20  PHE A  25  0 ...
            s_chain = line[21].strip()
            if s_chain != chain:
                continue
            try:
                start = int(line[22:26].strip())
                end   = int(line[33:37].strip())
                for rn in range(start, end + 1):
                    strand_set.add((rn, ""))
            except ValueError:
                pass

        elif line.startswith("SITE  "):
            # SITE records encode functional / active-site residues.
            # Format: cols 18-20 resname, 22 chain, 23-26 resnum  (up to 4 per line)
            offsets = [(18, 22, 23, 27), (29, 33, 34, 38),
                       (40, 44, 45, 49), (51, 55, 56, 60)]
            for rn_s, ch_s, seq_s, seq_e in offsets:
                try:
                    res_chain = line[ch_s].strip()
                    resnum    = int(line[seq_s:seq_e].strip())
                    if res_chain == chain:
                        active_site_set.add((resnum, ""))
                except (ValueError, IndexError):
                    pass

    residues_seen = {}
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        rec_chain = line[21].strip()
        if rec_chain != chain:
            continue
        resname   = line[17:20].strip()
        try:
            resnum  = int(line[22:26].strip())
        except ValueError:
            continue
        icode     = line[26].strip()
        atom_name = line[12:16].strip()
        record    = line[:6].strip()
        key       = (resnum, icode, record == "HETATM")

        if key not in residues_seen:
            residues_seen[key] = dict(
                chain=rec_chain, resnum=resnum, icode=icode, resname=resname,
                is_hetatm=(record == "HETATM"),
                ca_x=None, ca_y=None, ca_z=None,
                heavy_coords=[],
            )
        try:
            x = float(line[30:38]);  y_ = float(line[38:46]);  z = float(line[46:54])
        except ValueError:
            continue
        residues_seen[key]["heavy_coords"].append((x, y_, z))
        if atom_name == "CA":
            residues_seen[key]["ca_x"] = x
            residues_seen[key]["ca_y"] = y_
            residues_seen[key]["ca_z"] = z

    ligand_coords = []
    protein_res   = []
    for key, res in residues_seen.items():
        if res["is_hetatm"]:
            ligand_coords.extend(res["heavy_coords"])
        else:
            protein_res.append(res)

    binding_set = set()
    if ligand_coords:
        lig_xyz = np.array(ligand_coords)           # (L, 3)
        for res in protein_res:
            if not res["heavy_coords"]:
                continue
            prot_xyz = np.array(res["heavy_coords"])  # (A, 3)
            # Min distance between any protein atom and any ligand atom
            diff = prot_xyz[:, None, :] - lig_xyz[None, :, :]   # (A, L, 3)
            min_dist = np.sqrt((diff ** 2).sum(-1)).min()
            if min_dist <= 4.0:
                binding_set.add((res["resnum"], res["icode"]))

    ordered = sorted(
        [r for r in protein_res],
        key=lambda r: (r["resnum"], r["icode"])
    )
    for i, r in enumerate(ordered):
        r["idx"] = i

    return ordered, helix_set, strand_set, binding_set, active_site_set


def structural_role(resnum, icode, helix_set, strand_set, binding_set, active_site_set):
    """Return role string for coloring/labelling. Active site > binding > helix > strand > loop."""
    key = (resnum, icode)
    if key in active_site_set:
        return "active_site"
    if key in binding_set:
        return "binding_site"
    if key in helix_set:
        return "helix"
    if key in strand_set:
        return "strand"
    return "loop"


ROLE_COLOR = {
    "active_site":  "#ff6b35",   # orange-red
    "binding_site": "#e63946",   # red
    "helix":        "#457b9d",   # blue
    "strand":       "#2a9d8f",   # teal
    "loop":         "#adb5bd",   # grey
}
ROLE_LABEL = {
    "active_site":  "Active site (SITE record)",
    "binding_site": "Binding site (≤4 Å ligand)",
    "helix":        "Helix",
    "strand":       "Strand",
    "loop":         "Loop / coil",
}



ALL_ROLES = ["active_site", "binding_site", "helix", "strand", "loop"]


def compute_enrichment(df_protein, top_k=15):
    """
    For a single protein DataFrame (must have columns delta_alpha, structural_role),
    test whether the top-k |Δα| residues are enriched for each structural role.

    Returns a DataFrame with one row per role:
        role, n_top, n_bg, frac_top, frac_bg, odds_ratio, pvalue
    """
    df = df_protein.copy()
    df["abs_delta"] = df["delta_alpha"].abs()
    df_sorted = df.sort_values("abs_delta", ascending=False)
    top_set  = set(df_sorted.index[:top_k])
    N        = len(df)
    k        = min(top_k, N)
    rows = []
    for role in ALL_ROLES:
        role_idx  = set(df.index[df["structural_role"] == role])
        a = len(top_set & role_idx)                 # top & role
        b = len(top_set) - a                        # top & not-role
        c = len(role_idx) - a                       # not-top & role
        d = N - a - b - c                           # not-top & not-role
        if (a + b + c + d) == 0:
            continue
        _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        frac_top = a / max(k, 1)
        frac_bg  = len(role_idx) / max(N, 1)
        odds     = (a / max(b, 1)) / max(c / max(d, 1), 1e-9)
        rows.append(dict(role=role, n_top=a, n_background=len(role_idx),
                         frac_top=frac_top, frac_bg=frac_bg,
                         odds_ratio=odds, pvalue=pval))
    return pd.DataFrame(rows)


def compute_global_enrichment(all_rows, top_k=15):
    """Run enrichment on the pooled table (all proteins together)."""
    df = pd.DataFrame(all_rows)
    return compute_enrichment(df.reset_index(drop=True), top_k)


def plot_role_enrichment(enrich_df, out_path, title="Top-k residue enrichment by structural role"):
    """
    Side-by-side bar chart: fraction of top-k vs background for each role,
    with * markers for significant enrichment (p < 0.05).
    """
    roles     = enrich_df["role"].tolist()
    frac_top  = enrich_df["frac_top"].values
    frac_bg   = enrich_df["frac_bg"].values
    pvals     = enrich_df["pvalue"].values
    x = np.arange(len(roles))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_top = ax.bar(x - w/2, frac_top, w, label="Top-k |Δα|",
                      color=[ROLE_COLOR.get(r, "#999") for r in roles], alpha=0.9)
    bars_bg  = ax.bar(x + w/2, frac_bg,  w, label="Background",
                      color=[ROLE_COLOR.get(r, "#999") for r in roles], alpha=0.4,
                      hatch="//")

    # Annotate significant bars
    for i, (ft, fb, pv) in enumerate(zip(frac_top, frac_bg, pvals)):
        if pv < 0.05:
            marker = "***" if pv < 0.001 else ("**" if pv < 0.01 else "*")
            ax.text(i - w/2, ft + 0.005, marker, ha="center", fontsize=10, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([ROLE_LABEL.get(r, r) for r in roles], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of residues")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aggregate_delta_with_roles(a_s_all, a_m_all, roles_all, out_path, top_k=15):
    """
    Aggregate Δα across all proteins (aligned by ESM-IF residue index),
    colored by the modal structural role at each position.
    """
    d    = a_m_all - a_s_all
    cols = [ROLE_COLOR.get(r, "#adb5bd") for r in roles_all]

    fig, ax = plt.subplots(figsize=(2.5, 4))
    ax.bar(np.arange(len(d)), d, color=cols, width=1.0, linewidth=0)
    ax.axhline(0, color="black", lw=0.8, ls="--")

    # Mark top-k
    top_idx = np.argsort(np.abs(d))[-top_k:]
    for ti in top_idx:
        ax.axvline(ti, color="gold", lw=0.5, alpha=0.7, zorder=0)

    ax.set_xlabel("Residue index (ESM-IF order)")
    ax.set_ylabel("Δα  (multi − single)")
    ax.set_title("Aggregate Δα — all test proteins  (gold lines = top-k most differential)")

    patches = [mpatches.Patch(color=ROLE_COLOR[k], label=ROLE_LABEL[k]) for k in ROLE_COLOR]
    ax.legend(handles=patches, ncol=3, fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



@torch.no_grad()
def extract_attention_resvar(model, loader, device):
    results = []
    for batch in loader:
        esmif, res_mask, time_mask, _, y = batch
        esmif      = esmif.to(device).float()
        res_mask   = res_mask.to(device)
        time_mask  = time_mask.to(device)
        B, T, R, D = esmif.shape

        frame_valid = (~time_mask).float().unsqueeze(-1).unsqueeze(-1)
        res_valid   = res_mask.float().unsqueeze(-1)
        valid       = frame_valid * res_valid
        n_valid     = valid.sum(dim=1).clamp(min=1)

        x_masked = esmif * valid
        r_mean   = x_masked.sum(dim=1) / n_valid
        x2_mean  = (esmif ** 2 * valid).sum(dim=1) / n_valid
        r_std    = (x2_mean - r_mean ** 2).clamp(min=0).sqrt()

        if model.n_stats == 3:
            x_for_max = esmif + (1.0 - valid) * (-1e4)
            r_max, _  = x_for_max.max(dim=1)
            r_max     = r_max * (n_valid > 1).float()
            feat      = torch.cat([r_mean, r_std, r_max], dim=-1)
        else:
            feat = torch.cat([r_mean, r_std], dim=-1)

        res_global_valid = (n_valid.squeeze(-1) > 0)
        h      = model.res_norm(model.input_proj(feat))
        scores = model.score_net(h).squeeze(-1)
        scores = scores.masked_fill(~res_global_valid, float("-inf"))
        alpha  = torch.softmax(scores, dim=-1).cpu().numpy()

        for b in range(B):
            results.append((alpha[b], float(y[b])))
    return results


@torch.no_grad()
def extract_attention_transformer(model, loader, device, max_res=None):
    results = []
    for batch in loader:
        esmif, res_mask, time_mask, frame_idxs, y = batch
        esmif      = esmif.to(device).float()
        res_mask   = res_mask.to(device)
        time_mask  = time_mask.to(device)
        frame_idxs = frame_idxs.to(device)
        B, T, R, _ = esmif.shape

        x = esmif.view(B * T, R, -1)
        m = res_mask.view(B * T, R)
        x = model.res_norm(model.input_proj(x))
        scores = model.score_net(x).squeeze(-1).masked_fill(~m, float("-inf"))
        alpha  = torch.softmax(scores, dim=-1)

        alpha_bt   = alpha.view(B, T, R)
        valid_f    = (~time_mask).float().unsqueeze(-1)
        n_valid    = valid_f.sum(1).clamp(min=1)
        alpha_mean = (alpha_bt * valid_f).sum(1) / n_valid
        alpha_mean = alpha_mean.cpu().numpy()

        pad_to = max_res if max_res is not None else R
        for b in range(B):
            a = alpha_mean[b]  # shape (R,)
            if len(a) < pad_to:
                a = np.concatenate([a, np.zeros(pad_to - len(a))])
            results.append((a[:pad_to], float(y[b])))
    return results


@torch.no_grad()
def extract_predictions(model, loader, device):
    """Full forward pass → list of (y_true, p_pred) aligned with loader order."""
    results = []
    for batch in loader:
        esmif, res_mask, time_mask, frame_idxs, y = batch
        esmif      = esmif.to(device).float()
        res_mask   = res_mask.to(device)
        time_mask  = time_mask.to(device)
        frame_idxs = frame_idxs.to(device)
        logits = model(esmif, res_mask, time_mask, frame_idxs)
        probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        for b in range(len(y)):
            results.append((float(y[b]), float(probs[b])))
    return results



def plot_attention_profile(residues, a_s, a_m, roles, top_k, out_path, pdb_id):
    """
    Linear attention profile colored by structural role.
    Three panels: single-state α, multi-state α, Δα.
    """
    R    = len(residues)
    x    = np.arange(R)
    d    = a_m - a_s
    cols = [ROLE_COLOR[roles[i]] for i in range(R)]

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7), sharex=True)

    for val, ax, title in [(a_s, axes[0], f"{pdb_id} — Single-state (α)"),
                            (a_m, axes[1], f"{pdb_id} — Multi-state (α)")]:
        ax.bar(x, val, color=cols, width=1.0, linewidth=0)
        ax.set_ylabel("Mean α", fontsize=8);  ax.set_title(title, fontsize=9)

    # Δα: red positive, blue negative, still role-bordered
    d_cols = ["tomato" if v > 0 else "steelblue" for v in d]
    axes[2].bar(x, d, color=d_cols, width=1.0, linewidth=0)
    axes[2].axhline(0, color="black", lw=0.8, ls="--")
    axes[2].set_xlabel("Residue (ESM-IF order)", fontsize=8)
    axes[2].set_ylabel("Δα (multi − single)", fontsize=8)
    axes[2].set_title(f"{pdb_id} — Δα  (red = more attended in multi-state)", fontsize=9)

    # Annotate top-k Δα residues
    top_idx = np.argsort(np.abs(d))[-top_k:]
    for ti in sorted(top_idx):
        res    = residues[ti]
        role   = roles[ti]
        role_s = {"active_site": "AS", "binding_site": "BS", "helix": "H", "strand": "S", "loop": "L"}.get(role, role[:2].upper())
        label  = f"{res['resname']}{res['resnum']}({role_s})"
        axes[2].annotate(label, (ti, d[ti]),
                         textcoords="offset points",
                         xytext=(0, 5 if d[ti] >= 0 else -12),
                         ha="center", fontsize=5, rotation=90,
                         color=ROLE_COLOR[role])

    # PDB resnum x-tick every 20
    step   = max(1, R // 15)
    ticks  = x[::step]
    tlbls  = [str(residues[int(i)]["resnum"]) for i in ticks if int(i) < R]
    axes[0].set_xticks(ticks);  axes[0].set_xticklabels(tlbls, fontsize=6)

    # Legend
    patches = [mpatches.Patch(color=ROLE_COLOR[k], label=ROLE_LABEL[k])
               for k in ROLE_COLOR]
    axes[0].legend(handles=patches, ncol=3, fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_traj_attention(ax, residues, alpha, roles, top_k, label, color):
    """
    Draw a single-trajectory attention bar chart onto `ax`.
    Bars are colored by structural role; top-k residues are annotated.
    `label` appears as the y-axis title; `color` tints the background lightly.
    """
    R    = len(residues)
    x    = np.arange(R)
    cols = [ROLE_COLOR[roles[i]] for i in range(R)]

    ax.bar(x, alpha[:R], color=cols, width=1.0, linewidth=0)
    ax.set_facecolor((*matplotlib.colors.to_rgb(color), 0.04))
    ax.set_ylabel(label, fontsize=7, labelpad=2)
    ax.set_ylim(0, None)

    top_idx = np.argsort(alpha[:R])[-top_k:]
    for ti in sorted(top_idx):
        res  = residues[ti]
        role = roles[ti]
        rs   = {"active_site": "AS", "binding_site": "BS",
                "helix": "H", "strand": "S", "loop": "L"}.get(role, role[:2].upper())
        ax.annotate(f"{res['resname']}{res['resnum']}({rs})",
                    (ti, alpha[ti]),
                    textcoords="offset points", xytext=(0, 3),
                    fontsize=5, rotation=90, ha="center", va="bottom")


def plot_top_confident(samples, out_path, top_k=10):
    """
    samples: list of dicts with keys:
        residues, alpha, roles, prob, y_true, traj_id, pdb_id
    Sorted: first 5 = single-state (most confident), next 5 = multi-state.
    Produces a 10-row figure (or fewer if < 10 samples available).
    """
    n     = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(3.5, 1.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, samples):
        cls_name = "multi-state" if s["y_true"] == 1 else "single-state"
        col      = "tomato" if s["y_true"] == 1 else "steelblue"
        label    = (f"{s['pdb_id']}  [{cls_name}]  p={s['prob']:.3f}\n"
                    f"{s['traj_id']}")
        plot_single_traj_attention(ax, s["residues"], s["alpha"], s["roles"],
                                   top_k=top_k, label=label, color=col)

    # Shared role legend on the last axis
    patches = [mpatches.Patch(color=ROLE_COLOR[k], label=ROLE_LABEL[k]) for k in ROLE_COLOR]
    axes[-1].legend(handles=patches, ncol=len(ROLE_COLOR), fontsize=7,
                    loc="upper right", bbox_to_anchor=(1, -0.15))
    axes[-1].set_xlabel("Residue index (ESM-IF order)")

    fig.suptitle("Top-5 most confident trajectories per class — residue attention",
                 fontsize=11, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_structure_3views(residues, a_mean, roles, top_k, out_path, pdb_id):
    """
    Three 2D projections (XY, XZ, YZ) of Cα positions colored by attention.
    Only residues with valid Cα coordinates are shown.
    """
    valid = [(r, a_mean[r["idx"]], roles[r["idx"]])
             for r in residues
             if r["ca_x"] is not None and r["idx"] < len(a_mean)]
    if not valid:
        return

    xs    = np.array([r["ca_x"] for r, _, _ in valid])
    ys    = np.array([r["ca_y"] for r, _, _ in valid])
    zs    = np.array([r["ca_z"] for r, _, _ in valid])
    attn  = np.array([a           for _, a, _ in valid])
    roles_v = [role               for _, _, role in valid]
    recs  = [r                    for r, _, _ in valid]

    # Normalize attention to [0,1] for colormap
    attn_norm = (attn - attn.min()) / max(attn.max() - attn.min(), 1e-8)

    projections = [
        ("XY — top view",   xs, ys,  "X (Å)", "Y (Å)"),
        ("XZ — front view", xs, zs,  "X (Å)", "Z (Å)"),
        ("YZ — side view",  ys, zs,  "Y (Å)", "Z (Å)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = plt.cm.RdYlBu_r

    for ax, (title, px, py, xl, yl) in zip(axes, projections):
        sc = ax.scatter(px, py, c=attn_norm, cmap=cmap,
                        s=30, alpha=0.85, linewidths=0.3,
                        edgecolors="black", zorder=2)
        # Draw backbone as thin line
        ax.plot(px, py, color="lightgrey", lw=0.5, zorder=1)

        # Mark binding site residues with a ring
        bs_idx = [i for i, role in enumerate(roles_v) if role == "binding_site"]
        if bs_idx:
            ax.scatter(px[bs_idx], py[bs_idx],
                       s=90, facecolors="none", edgecolors="red",
                       linewidths=1.5, zorder=3, label="Binding site")

        # Label top-k by attention
        top_idx = np.argsort(attn)[-top_k:]
        for ti in top_idx:
            res    = recs[ti]
            role   = roles_v[ti]
            rs     = {"active_site": "AS", "binding_site": "BS", "helix": "H", "strand": "S", "loop": "L"}.get(role, role[:2].upper())
            ax.annotate(f"{res['resname']}{res['resnum']}({rs})",
                        (px[ti], py[ti]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color=ROLE_COLOR[role],
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))

        ax.set_xlabel(xl);  ax.set_ylabel(yl)
        ax.set_title(f"{pdb_id} — {title}")
        ax.set_aspect("equal")
        if bs_idx:
            ax.legend(fontsize=7)

    cbar = fig.colorbar(sc, ax=axes[-1], shrink=0.8)
    cbar.set_label("Attention weight (normalised)")

    # Role legend
    patches = [mpatches.Patch(color=ROLE_COLOR[k], label=ROLE_LABEL[k])
               for k in ROLE_COLOR]
    fig.legend(handles=patches, ncol=4, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Cα structure colored by mean attention — {pdb_id}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binding_site_zoom(residues, a_s, a_m, roles, out_path, pdb_id):
    """Bar chart focusing only on binding-site residues."""
    bs = [(r, a_s[r["idx"]], a_m[r["idx"]])
          for r in residues
          if r["idx"] < len(a_s) and roles[r["idx"]] == "binding_site"]
    if not bs:
        return

    labels = [f"{r['resname']}{r['resnum']}" for r, _, _ in bs]
    vals_s = np.array([s for _, s, _ in bs])
    vals_m = np.array([m for _, _, m in bs])
    x      = np.arange(len(bs))
    w      = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(bs) * 0.55 + 2), 5))

    axes[0].bar(x - w/2, vals_s, w, color="steelblue", label="Single-state")
    axes[0].bar(x + w/2, vals_m, w, color="tomato",    label="Multi-state")
    axes[0].set_xticks(x);  axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean α");  axes[0].set_title("Binding-site residue attention")
    axes[0].legend()

    delta = vals_m - vals_s
    cols  = ["tomato" if d > 0 else "steelblue" for d in delta]
    axes[1].bar(x, delta, color=cols)
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xticks(x);  axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Δα (multi − single)")
    axes[1].set_title("Δα at binding site  (red = more attended in multi-state)")

    fig.suptitle(f"{pdb_id} — Binding-site residues ({len(bs)} residues within 4 Å of ligand)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def plot_roc_curves(curves, out_path):
    """
    curves: list of (label, y_true_array, probs_array)
    Saves a single figure with one ROC curve per entry.
    """
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))
    for (label, y_true, probs), col in zip(curves, colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc     = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{label}  (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _load_model_from_ckpt(ckpt_path, model_type, device):
    """Load a model (resvar or transformer) from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["args"]
    if model_type == "resvar":
        from train_residue_variance import ResidueVarianceClassifier
        model = ResidueVarianceClassifier(
            d_inner    = cfg.get("d_inner",    128),
            d_model    = cfg.get("d_model",    128),
            mlp_hidden = cfg.get("mlp_hidden", 256),
            dropout    = 0.0,
            n_stats    = cfg.get("n_stats",    3),
        ).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        from model import ResAttnTransformer
        model = ResAttnTransformer(
            d_inner      = cfg.get("d_inner",      128),
            d_model      = cfg.get("d_model",      128),
            n_heads      = cfg.get("n_heads",        4),
            n_layers     = cfg.get("n_layers",       2),
            dim_ff       = cfg.get("dim_ff",       256),
            score_hidden = cfg.get("score_hidden",  32),
            layer_drop   = 0.0,
            causal       = cfg.get("causal",      False),
            dropout      = 0.0,
        ).to(device)
        state = ckpt["model"]
        if any(k.startswith("transformer.layers.") for k in state):
            state = {k.replace("transformer.layers.", "layers.", 1): v
                     for k, v in state.items()}
        model.load_state_dict(state)
    model.eval()
    return model, cfg



def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",         required=True)
    p.add_argument("--extra-checkpoints",  nargs="*", default=[],
                   metavar="CKPT",
                   help="Additional checkpoints to overlay on the ROC plot")
    p.add_argument("--extra-labels",       nargs="*", default=[],
                   metavar="LABEL",
                   help="Display names for --extra-checkpoints (same order)")
    p.add_argument("--test-csv",     default=None)
    p.add_argument("--meta-csv",     default="data/processed/protein_level_test.csv")
    p.add_argument("--local-cache",  default=None)
    p.add_argument("--pdb-dir",      default="/tmp/pdb_cache")
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--model-type",   default="resvar", choices=["resvar", "transformer"])
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--top-k",        type=int, default=15)
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    out_dir   = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "residue_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model type : {args.model_type}")
    print(f"Device     : {device}\n")

    model, cfg = _load_model_from_ckpt(args.checkpoint, args.model_type, device)

    test_csv    = args.test_csv    or cfg["test_csv"]
    local_cache = args.local_cache or cfg.get("local_cache")
    max_frames  = cfg.get("max_frames", 64)
    max_res     = cfg.get("max_res", 200)

    from train_mamba import ChunkDataset, collate_fn

    test_meta = pd.read_csv(test_csv)
    if "esmif_emb_file" in test_meta.columns:
        from live_dataset import LiveChunkDataset
        ds = LiveChunkDataset(test_meta, max_frames=max_frames, max_res=max_res)
    else:
        ds = ChunkDataset(test_meta, local_cache, max_frames=max_frames, max_res=max_res)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=True)

    print("Computing ROC curves ...")
    primary_preds = extract_predictions(model, loader, device)
    primary_label = Path(args.checkpoint).parent.name
    roc_curves = [(primary_label,
                   np.array([y for y, _ in primary_preds]),
                   np.array([p for _, p in primary_preds]))]

    extra_labels = list(args.extra_labels)
    for i, extra_ckpt in enumerate(args.extra_checkpoints):
        label = extra_labels[i] if i < len(extra_labels) else Path(extra_ckpt).parent.name
        print(f"  Loading extra checkpoint: {extra_ckpt}  ({label})")
        extra_model, _ = _load_model_from_ckpt(extra_ckpt, args.model_type, device)
        extra_preds     = extract_predictions(extra_model, loader, device)
        roc_curves.append((label,
                           np.array([y for y, _ in extra_preds]),
                           np.array([p for _, p in extra_preds])))

    plot_roc_curves(roc_curves, out_dir / "roc_curves.png")

    if args.model_type == "resvar":
        print("Extracting residue attention (ResidueVarianceClassifier) ...")
        attn_results = extract_attention_resvar(model, loader, device)
    else:
        print("Extracting residue attention (ResAttnTransformer Stage 1) ...")
        attn_results = extract_attention_transformer(
            model, loader, device, max_res=cfg.get("max_res", None)
        )

    print(f"  Collected attention for {len(attn_results)} samples\n")

    # Pad to uniform length in case proteins have different residue counts
    _max_r = max(len(a) for a, _ in attn_results)
    def _pad(a): return np.concatenate([a, np.zeros(_max_r - len(a))]) if len(a) < _max_r else a
    alphas = np.array([_pad(a) for a, _ in attn_results])
    ys     = np.array([y for _, y in attn_results])

    if "receptor" in test_meta.columns:
        receptors = test_meta["receptor"].values
    elif Path(args.meta_csv).exists():
        meta   = pd.read_csv(args.meta_csv)
        merged = test_meta.merge(meta[["traj_id", "receptor"]], on="traj_id", how="left")
        receptors = merged["receptor"].fillna("unknown").values
    else:
        receptors = np.array([f"unknown_{i}" for i in range(len(test_meta))])

    receptor_records = defaultdict(lambda: {"alphas_single": [], "alphas_multi": []})
    for i, rec in enumerate(receptors):
        key = "alphas_multi" if ys[i] == 1 else "alphas_single"
        receptor_records[rec][key].append(alphas[i])

    a_s_all = alphas[ys == 0].mean(0) if (ys == 0).any() else np.zeros(alphas.shape[1])
    a_m_all = alphas[ys == 1].mean(0) if (ys == 1).any() else np.zeros(alphas.shape[1])
    d_all   = a_m_all - a_s_all
    cols    = ["tomato" if v > 0 else "steelblue" for v in d_all]

    fig, ax = plt.subplots(figsize=(2.5, 4))
    ax.bar(np.arange(len(d_all)), d_all, color=cols, width=1.0)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Residue index", fontsize=7);  ax.set_ylabel("Δα", fontsize=7)
    ax.set_title("Aggregate Δα — all test samples pooled", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.set_size_inches(2.5, 4)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_delta_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved aggregate_delta_alpha.png")

    # Reuse predictions already computed for ROC (primary_preds)
    probs = np.array([p for _, p in primary_preds])     # predicted P(multi-state)
    conf  = np.where(ys == 1, probs, 1.0 - probs)       # confidence = P(correct class)

    traj_ids   = test_meta["traj_id"].values if "traj_id" in test_meta.columns \
                 else np.array([f"sample_{i}" for i in range(len(ys))])
    chunk_col  = next((c for c in ["chunk_id", "chunk", "chunk_idx"] if c in test_meta.columns), None)
    chunk_vals = test_meta[chunk_col].values if chunk_col else None

    top5_single = np.argsort(conf[ys == 0])[::-1][:5]
    top5_multi  = np.argsort(conf[ys == 1])[::-1][:5]
    idx_single  = np.where(ys == 0)[0][top5_single]
    idx_multi   = np.where(ys == 1)[0][top5_multi]

    print("\nTop-5 most confident SINGLE-STATE predictions:")
    for rank, i in enumerate(idx_single, 1):
        chunk_str = f"  chunk={chunk_vals[i]}" if chunk_vals is not None else ""
        print(f"  {rank}. {traj_ids[i]}{chunk_str}  |  p(multi)={probs[i]:.4f}  conf={conf[i]:.4f}")

    print("\nTop-5 most confident MULTI-STATE predictions:")
    for rank, i in enumerate(idx_multi, 1):
        chunk_str = f"  chunk={chunk_vals[i]}" if chunk_vals is not None else ""
        print(f"  {rank}. {traj_ids[i]}{chunk_str}  |  p(multi)={probs[i]:.4f}  conf={conf[i]:.4f}")
    print()

    # Cache PDB parses so we don't re-download within this loop
    _pdb_cache = {}

    def _get_pdb_info(rec):
        if rec in _pdb_cache:
            return _pdb_cache[rec]
        pdb_id, chain = parse_pdb_id_chain(rec)
        if pdb_id is None:
            _pdb_cache[rec] = None; return None
        pdb_path = download_pdb(pdb_id, args.pdb_dir)
        if pdb_path is None:
            _pdb_cache[rec] = None; return None
        residues, helix_set, strand_set, binding_set, active_site_set = \
            parse_pdb(pdb_path, chain)
        if not residues:
            _pdb_cache[rec] = None; return None
        _pdb_cache[rec] = (pdb_id, residues, helix_set, strand_set, binding_set, active_site_set)
        return _pdb_cache[rec]

    top_samples = []
    for label_order, idx_list in [("single", idx_single), ("multi", idx_multi)]:
        for i in idx_list:
            rec   = receptors[i]
            info  = _get_pdb_info(rec)
            if info is None:
                continue
            pdb_id, residues, helix_set, strand_set, binding_set, active_site_set = info
            R_use = min(len(residues), alphas.shape[1])
            roles = [structural_role(r["resnum"], r["icode"],
                                     helix_set, strand_set, binding_set, active_site_set)
                     for r in residues[:R_use]]
            top_samples.append(dict(
                residues = residues[:R_use],
                alpha    = alphas[i, :R_use],
                roles    = roles,
                prob     = float(probs[i]),
                y_true   = int(ys[i]),
                traj_id  = str(traj_ids[i]),
                pdb_id   = pdb_id,
            ))

    if top_samples:
        out_conf = out_dir / "top_confident_trajectories.png"
        plot_top_confident(top_samples, out_conf, top_k=args.top_k)
        print(f"Saved top_confident_trajectories.png  ({len(top_samples)} panels)")

    all_rows = []
    for rec in [r for r in receptor_records if not r.startswith("unknown")]:
        pdb_id, chain = parse_pdb_id_chain(rec)
        if pdb_id is None:
            continue

        print(f"\n{rec}  (PDB={pdb_id}, chain={chain})")
        pdb_path = download_pdb(pdb_id, args.pdb_dir)
        if pdb_path is None:
            print(f"  ✗ Download failed for {pdb_id}")
            continue

        residues, helix_set, strand_set, binding_set, active_site_set = \
            parse_pdb(pdb_path, chain)
        if not residues:
            print(f"  ✗ No residues found in chain {chain}")
            continue

        R_pdb = len(residues)
        R_emb = alphas.shape[1]
        R_use = min(R_pdb, R_emb)
        print(f"  {R_pdb} residues in PDB | {R_emb} in embedding | using {R_use}")
        print(f"  Active-site: {len(active_site_set)} | "
              f"Binding-site: {len(binding_set)} | "
              f"Helix: {len(helix_set)} | Strand: {len(strand_set)}")

        rec_data = receptor_records[rec]
        arr_s    = np.array(rec_data["alphas_single"]) if rec_data["alphas_single"] else None
        arr_m    = np.array(rec_data["alphas_multi"])  if rec_data["alphas_multi"]  else None
        a_s = arr_s[:, :R_use].mean(0) if arr_s is not None else np.zeros(R_use)
        a_m = arr_m[:, :R_use].mean(0) if arr_m is not None else np.zeros(R_use)
        a_mean = (a_s + a_m) / 2

        roles = [structural_role(r["resnum"], r["icode"],
                                 helix_set, strand_set, binding_set, active_site_set)
                 for r in residues[:R_use]]

        # Output directory per receptor
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", rec)
        rec_dir = out_dir / "per_receptor" / safe
        rec_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: attention profile
        plot_attention_profile(residues[:R_use], a_s, a_m, roles,
                               args.top_k,
                               rec_dir / "01_attention_profile.png", pdb_id)
        print(f"  Saved 01_attention_profile.png")

        # Plot 2: 3D structure 3 views
        plot_structure_3views(residues[:R_use], a_mean, roles,
                              args.top_k,
                              rec_dir / "02_structure_3views.png", pdb_id)
        print(f"  Saved 02_structure_3views.png")

        # Plot 3: binding site zoom
        plot_binding_site_zoom(residues[:R_use], a_s, a_m, roles,
                               rec_dir / "03_binding_site_zoom.png", pdb_id)
        print(f"  Saved 03_binding_site_zoom.png")

        # Rows for CSV
        protein_rows = []
        for res in residues[:R_use]:
            i = res["idx"]
            row = dict(
                receptor=rec, pdb_id=pdb_id, chain=res["chain"],
                resnum=res["resnum"], resname=res["resname"],
                residue_idx=i,
                structural_role=roles[i],
                mean_alpha_single=float(a_s[i]),
                mean_alpha_multi=float(a_m[i]),
                delta_alpha=float(a_m[i] - a_s[i]),
                ca_x=res["ca_x"], ca_y=res["ca_y"], ca_z=res["ca_z"],
            )
            protein_rows.append(row)
            all_rows.append(row)

        # Per-protein enrichment
        if protein_rows:
            df_p = pd.DataFrame(protein_rows)
            enrich_p = compute_enrichment(df_p.reset_index(drop=True), args.top_k)
            if not enrich_p.empty:
                enrich_p.to_csv(rec_dir / "enrichment.csv", index=False)
                plot_role_enrichment(enrich_p,
                                     rec_dir / "04_role_enrichment.png",
                                     title=f"{pdb_id} — Top-{args.top_k} |Δα| role enrichment")
                print(f"  Saved 04_role_enrichment.png")
                sig = enrich_p[enrich_p["pvalue"] < 0.05]
                if not sig.empty:
                    print(f"  Significant enrichments (p<0.05):")
                    for _, row in sig.iterrows():
                        print(f"    {row['role']}: {row['n_top']}/{args.top_k} top "
                              f"vs {row['frac_bg']:.1%} background, "
                              f"OR={row['odds_ratio']:.1f}, p={row['pvalue']:.3f}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(out_dir / "attention_by_residue.csv", index=False)
        print(f"\nSaved attention_by_residue.csv  "
              f"({len(df)} residues, {df['pdb_id'].nunique()} structures)")

        print("\n" + "=" * 60)
        print("GLOBAL ENRICHMENT: Top-k |Δα| residues vs background")
        print(f"  (top-k = {args.top_k} per protein, aggregated across all proteins)")
        print("=" * 60)

        enrich_global = compute_enrichment(df.reset_index(drop=True), args.top_k)
        enrich_global.to_csv(out_dir / "global_enrichment.csv", index=False)

        # Pretty print
        print(f"\n{'Role':<20} {'n_top':>6} {'frac_top':>9} {'frac_bg':>8} "
              f"{'OR':>6} {'p-value':>9}  sig")
        print("-" * 65)
        for _, row in enrich_global.sort_values("pvalue").iterrows():
            sig = ("***" if row["pvalue"] < 0.001 else
                   "**"  if row["pvalue"] < 0.01  else
                   "*"   if row["pvalue"] < 0.05  else "")
            print(f"  {row['role']:<18} {int(row['n_top']):>6} "
                  f"{row['frac_top']:>8.1%} {row['frac_bg']:>8.1%} "
                  f"{row['odds_ratio']:>6.2f} {row['pvalue']:>9.4f}  {sig}")

        plot_role_enrichment(enrich_global,
                             out_dir / "global_role_enrichment.png",
                             title=f"Global top-{args.top_k} |Δα| enrichment by structural role")
        print("\nSaved global_role_enrichment.png")

        print("\nMean |Δα| by structural role:")
        df["abs_delta"] = df["delta_alpha"].abs()
        role_summary = (df.groupby("structural_role")["abs_delta"]
                          .agg(mean="mean", std="std", count="count")
                          .sort_values("mean", ascending=False))
        print(role_summary.to_string())

        print(f"\nTop-{args.top_k} most differential residues (by |Δα|, pooled):")
        top_global = (df.nlargest(args.top_k, "abs_delta")
                        [["pdb_id", "resname", "resnum", "structural_role",
                          "delta_alpha", "mean_alpha_multi", "mean_alpha_single"]])
        print(top_global.to_string(index=False))
        top_global.to_csv(out_dir / "top_k_residues.csv", index=False)

        # Build aggregate using ESM-IF residue index aligned across all samples
        a_s_all = alphas[ys == 0].mean(0) if (ys == 0).any() else np.zeros(alphas.shape[1])
        a_m_all = alphas[ys == 1].mean(0) if (ys == 1).any() else np.zeros(alphas.shape[1])
        # Modal role per residue index
        if "residue_idx" in df.columns:
            modal_roles = (df.groupby("residue_idx")["structural_role"]
                             .agg(lambda x: x.value_counts().index[0]))
            roles_all = [modal_roles.get(i, "loop") for i in range(len(a_s_all))]
        else:
            roles_all = ["loop"] * len(a_s_all)
        plot_aggregate_delta_with_roles(a_s_all, a_m_all, roles_all,
                                        out_dir / "aggregate_delta_alpha_roles.png",
                                        top_k=args.top_k)
        print("Saved aggregate_delta_alpha_roles.png")

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
