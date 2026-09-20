"""
Why Layer Norm? — A Visual Explanation
========================================
This animation shows the REAL problem LN solves in neural networks:

Scene Group 1: "The Problem" — Without LayerNorm
   Different layers produce wildly different output distributions.
   Some neurons explode, some vanish. Training becomes unstable.

Scene Group 2: "The Solution" — With LayerNorm
   After LN, every layer's output is nicely normalized.
   Stable, consistent distributions across all layers.

Scene Group 3: "The Pipeline" — Step-by-step on a feature vector
   Take one sample's feature vector (50 dims) and show
   the 2 core steps: center (subtract mean) -> normalize (divide by std),
   ending at mean~0, std~1.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('Agg')

np.random.seed(42)

# ============================================================
# Simulate feature vectors at different "layers" of a network
# ============================================================
n_features = 50  # feature dimension (like d_model in Transformer)
n_samples = 5    # show a few sample vectors

# Simulate different layers having wildly different distributions
layer_configs = [
    {"name": "Layer 1 output",  "mean": 0.5,  "std": 0.3},
    {"name": "Layer 4 output",  "mean": 2.5,  "std": 1.8},
    {"name": "Layer 8 output",  "mean": -1.2, "std": 0.1},
    {"name": "Layer 12 output", "mean": 8.0,  "std": 5.0},
    {"name": "Layer 16 output", "mean": -4.0, "std": 3.0},
]

# Generate raw feature vectors for each "layer"
raw_features = []
for cfg in layer_configs:
    feat = np.random.randn(n_features) * cfg["std"] + cfg["mean"]
    raw_features.append(feat)

# Apply LayerNorm to each
eps = 1e-5
gamma = np.ones(n_features) * 1.0  # for simplicity
beta = np.zeros(n_features)

def layer_norm(x):
    mu = x.mean()
    sigma = x.std() + eps
    return (x - mu) / sigma * gamma + beta

normed_features = [layer_norm(f) for f in raw_features]

# ============================================================
# One detailed sample for the step-by-step pipeline
# ============================================================
sample_feat = raw_features[3].copy()  # use Layer 12 (most dramatic)
pipe_step0 = sample_feat.copy()
pipe_step1 = sample_feat - sample_feat.mean()
pipe_step2 = (sample_feat - sample_feat.mean()) / (sample_feat.std() + eps)
# Final result: mean~0, std~1 (standard LayerNorm without learnable params)

# ============================================================
# Create the animation figure: 3 scene groups
# ============================================================
# Scene timing
frames_per_scene = 80
pause_frames = 20
transition_frames = frames_per_scene - pause_frames
extra_end_pause = 40  # extra pause frames at the very end so last step fully lands

# --- Color scheme: color is tied to the LAYER, consistent across ALL groups ---
# Layer 1 = blue, Layer 4 = green, Layer 8 = orange, Layer 12 = purple, Layer 16 = red
layer_colors = [
    "#4FC3F7",  # Layer 1  — light blue
    "#66BB6A",  # Layer 4  — green
    "#FFB74D",  # Layer 8  — orange
    "#CE93D8",  # Layer 12 — purple
    "#EF5350",  # Layer 16 — red
]

PIPE_LAYER_IDX = 3                      # pipeline demos Layer 12
pipe_color = layer_colors[PIPE_LAYER_IDX]  # so pipeline uses Layer 12's purple

# Define all scenes as a list
scenes = []

# --- Scene group 1: WITHOUT LayerNorm (5 layers shown one by one) ---
for i, cfg in enumerate(layer_configs):
    scenes.append({
        "type": "bar_compare",
        "title": f'Without LayerNorm: {cfg["name"]}',
        "subtitle": f'mean={cfg["mean"]:.1f}, std={cfg["std"]:.1f} — Distribution shifts per layer!',
        "data_from": raw_features[max(0, i-1)] if i > 0 else np.zeros(n_features),
        "data_to": raw_features[i],
        "color": layer_colors[i],
        "ylim_from": (-15, 15),
        "ylim_to": (-15, 15),
        "badge": "NO LAYER NORM",
        "badge_color": "#EF5350",
    })

# --- Transition: Group1 last → Group2 first (raw Layer16 → normed Layer1) ---
scenes.append({
    "type": "bar_compare",
    "title": "Now applying LayerNorm...",
    "subtitle": "Watch how distributions become stable!",
    "data_from": raw_features[-1],       # last raw layer (Layer 16)
    "data_to": normed_features[0],       # first normed layer (Layer 1)
    "color": layer_colors[0],            # Layer 1's blue
    "ylim_from": (-15, 15),              # start with wide range
    "ylim_to": (-4, 4),                  # smoothly shrink
    "badge": "WITH LAYER NORM",
    "badge_color": "#66BB6A",
})

# --- Scene group 2: WITH LayerNorm (layers 2-5, layer 1 already in transition) ---
for i, cfg in enumerate(layer_configs):
    if i == 0:
        continue
    scenes.append({
        "type": "bar_compare",
        "title": f'With LayerNorm: {cfg["name"]}',
        "subtitle": "After LN: mean~0, std~1 — Stable across ALL layers!",
        "data_from": normed_features[i-1],
        "data_to": normed_features[i],
        "color": layer_colors[i],        # same layer => same color as Group 1
        "ylim_from": (-4, 4),
        "ylim_to": (-4, 4),
        "badge": "WITH LAYER NORM",
        "badge_color": "#66BB6A",
    })

# --- Transition: Group2 last → Group3 first (normed Layer16 → raw Layer12 for pipeline) ---
scenes.append({
    "type": "bar_compare",
    "title": "Pipeline Demo: Start with Layer 12 raw output",
    "subtitle": "Let's see LayerNorm step by step...",
    "data_from": normed_features[-1],    # last normed layer
    "data_to": pipe_step0,               # raw Layer 12
    "color": pipe_color,                 # Layer 12's purple
    "ylim_from": (-4, 4),               # start small
    "ylim_to": (-15, 20),               # expand to fit raw data
    "badge": "PIPELINE (Layer 12)",
    "badge_color": pipe_color,
})

# --- Scene group 3: Pipeline steps (Layer 12's color throughout) ---
pipe_stages = [
    ("Step 1: Subtract Mean (x - mu)",
     f"mean: {pipe_step0.mean():.1f} -> 0.0  |  Centering the distribution",
     pipe_step0, pipe_step1, (-15, 20), (-15, 15)),
    ("Step 2: Divide by Std (x / sigma)",
     f"std: {pipe_step0.std():.1f} -> 1.0  |  Normalizing variance. Done!",
     pipe_step1, pipe_step2, (-15, 15), (-4, 4)),
]
for title, subtitle, data_from, data_to, ylim_from, ylim_to in pipe_stages:
    scenes.append({
        "type": "bar_compare",
        "title": title,
        "subtitle": subtitle,
        "data_from": data_from,
        "data_to": data_to,
        "color": pipe_color,             # Layer 12's purple, consistent
        "ylim_from": ylim_from,
        "ylim_to": ylim_to,
        "badge": "PIPELINE (Layer 12)",
        "badge_color": pipe_color,
    })

n_scenes_total = len(scenes)
total_frames = n_scenes_total * frames_per_scene + extra_end_pause

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='#0d1117')
ax.set_facecolor('#0d1117')

# Pre-create bar objects
bar_x = np.arange(n_features)
bars = ax.bar(bar_x, np.zeros(n_features), color='#4FC3F7', alpha=0.85,
              edgecolor='white', linewidth=0.3, zorder=3)

# Zero line
ax.axhline(y=0, color='#666666', linewidth=1.0, linestyle='-', zorder=2)

# Text
title_text = ax.set_title("", fontsize=14, color='white', pad=15, fontweight='bold')

subtitle_text = ax.text(0.5, 0.95, "", transform=ax.transAxes,
                        fontsize=10, color='#aaaaaa',
                        horizontalalignment='center', verticalalignment='top',
                        zorder=10)

badge_obj = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color='#66BB6A',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                              edgecolor='#66BB6A', alpha=0.9),
                    zorder=10)

stats_obj = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                    fontsize=9, color='#80cbc4',
                    fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                              edgecolor='#80cbc4', alpha=0.85),
                    zorder=10)

progress_obj = ax.text(0.5, 0.02, "", transform=ax.transAxes,
                       fontsize=9, color='#666',
                       horizontalalignment='center', verticalalignment='bottom',
                       zorder=10)

ax.set_xlabel("Feature dimension index", fontsize=10, color='#888888')
ax.set_ylabel("Activation value", fontsize=10, color='#888888')
ax.tick_params(colors='#555555', labelsize=8)
for spine in ax.spines.values():
    spine.set_color('#333333')
ax.set_xlim(-1, n_features)

# ============================================================
# Smoothstep
# ============================================================
def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

# ============================================================
# Update
# ============================================================
def update(frame):
    scene_idx = frame // frames_per_scene
    frame_in_scene = frame % frames_per_scene

    # Clamp to last scene for extra end-pause frames
    if scene_idx >= n_scenes_total:
        scene_idx = n_scenes_total - 1
        frame_in_scene = frames_per_scene - 1  # hold at final state

    sc = scenes[scene_idx]

    # Interpolation
    if frame_in_scene < transition_frames:
        raw_t = frame_in_scene / transition_frames
        t = smoothstep(raw_t)
    else:
        t = 1.0

    data_from = sc["data_from"]
    data_to = sc["data_to"]
    current_data = data_from + (data_to - data_from) * t

    # Update bars
    for i, bar in enumerate(bars):
        val = current_data[i]
        bar.set_height(val)
        bar.set_y(min(0, val))
        bar.set_height(abs(val))
        if val >= 0:
            bar.set_y(0)
        else:
            bar.set_y(val)
        bar.set_color(sc["color"])
        bar.set_alpha(0.85)

    # Y limits — smooth interpolation
    ylim_from = sc["ylim_from"]
    ylim_to = sc["ylim_to"]
    cur_ymin = ylim_from[0] + (ylim_to[0] - ylim_from[0]) * t
    cur_ymax = ylim_from[1] + (ylim_to[1] - ylim_from[1]) * t
    ax.set_ylim(cur_ymin, cur_ymax)

    # Text
    title_text.set_text(sc["title"])
    title_text.set_color(sc["color"])
    subtitle_text.set_text(sc["subtitle"])

    badge_obj.set_text(sc["badge"])
    badge_obj.set_color(sc["badge_color"])
    badge_obj.get_bbox_patch().set_edgecolor(sc["badge_color"])

    cur_mean = current_data.mean()
    cur_std = current_data.std()
    stats_obj.set_text(f"mean={cur_mean:+.2f}  std={cur_std:.2f}")

    # Progress
    progress_parts = []
    for i in range(n_scenes_total):
        if i < scene_idx:
            progress_parts.append("●")
        elif i == scene_idx:
            progress_parts.append("◉")
        else:
            progress_parts.append("○")
    # Group visually: Group1(5) | trans+Group2(5) | trans+Group3(3) = 13
    g1 = " ".join(progress_parts[:5])
    g2 = " ".join(progress_parts[5:10])
    g3 = " ".join(progress_parts[10:])
    progress_obj.set_text(
        f'[{scene_idx+1}/{n_scenes_total}] {g1} | {g2} | {g3}'
    )

    return list(bars) + [title_text, subtitle_text, badge_obj, stats_obj, progress_obj]


# ============================================================
# Generate
# ============================================================
print(f"LayerNorm explainer: {n_scenes_total} scenes, {total_frames} frames")
print("Generating animation...")

anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

output_path = "/data/workspace/transformer/layernorm_animation.gif"
anim.save(output_path, writer='pillow', fps=20, dpi=100)
print(f"Animation saved: {output_path}")
plt.close(fig)

# ============================================================
# Keyframes — one per scene group (3 groups now)
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0d1117')

group_titles = [
    "Without LayerNorm: Each layer has different distribution",
    "With LayerNorm: All layers normalized to mean~0, std~1",
    "Pipeline: center -> normalize (mean=0, std=1)",
]
# Scene indices: Group1=[0..4], trans=5, Group2=[6..9], trans=10, Group3=[11..12]
group_indices = [
    list(range(0, 5)),              # 5 raw layers
    [5] + list(range(6, 10)),       # transition + 4 normed layers = 5
    [10] + list(range(11, 13)),     # transition + 2 pipeline steps = 3
]
group_title_colors = ["#EF5350", "#66BB6A", pipe_color]

for ax_idx, ax2 in enumerate(axes2.flatten()):
    ax2.set_facecolor('#0d1117')
    ax2.tick_params(colors='#555555', labelsize=7)
    for spine in ax2.spines.values():
        spine.set_color('#333333')

    indices = group_indices[ax_idx]
    ax2.set_title(group_titles[ax_idx], fontsize=10, color=group_title_colors[ax_idx],
                  fontweight='bold', pad=10)

    # Overlay all final states in this group
    alphas = np.linspace(0.3, 0.9, len(indices))
    for j, si in enumerate(indices):
        sc = scenes[si]
        data = sc["data_to"]
        label_short = sc["title"].split(":")[0] if ":" in sc["title"] else sc["title"]
        ax2.bar(bar_x + j * 0.15 - len(indices) * 0.075,
                data, width=0.15, alpha=alphas[j],
                color=sc["color"], label=label_short[:25],
                edgecolor='none')

    ax2.axhline(y=0, color='#666666', linewidth=0.8)
    ax2.set_xlim(-1, n_features)
    ax2.set_xlabel("Feature dim", fontsize=8, color='#666')
    ax2.set_ylabel("Value", fontsize=8, color='#666')
    ax2.legend(fontsize=6, loc='upper right', framealpha=0.5,
               facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

fig2.suptitle("Why LayerNorm? — Visual Explanation",
              fontsize=16, color='white', fontweight='bold', y=1.01)
plt.tight_layout()

keyframe_path = "/data/workspace/transformer/layernorm_keyframes.png"
fig2.savefig(keyframe_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"Keyframes saved: {keyframe_path}")

plt.close('all')
print("Done!")
