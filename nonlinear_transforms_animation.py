"""
2D Nonlinear Transforms Animation
===================================
展示二维空间中多种经典非线性变换，每种变换都作用在同一组正态分布点和网格上，
坐标系（坐标轴范围）始终固定不变。

与线性变换的关键区别：
  - 线性变换：直线映射为直线，网格保持平行
  - 非线性变换：直线可能弯曲、网格会扭曲变形

包含的非线性变换：
  1.  ReLU (Rectified Linear Unit)
  2.  Sigmoid (Logistic)
  3.  Tanh (Hyperbolic Tangent)
  4.  Swirl / Vortex (Radial-dependent rotation)
  5.  Radial Squeeze (Squash radius nonlinearly)
  6.  Polar Warp (Cartesian -> Polar mapping)
  7.  Sinusoidal Ripple
  8.  Softplus
  9.  Exponential Mapping
  10. Fisheye / Barrel Distortion
  11. Fold (absolute value)
  12. Quadratic (element-wise square with sign)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

# ============================================================
# Generate data — 2D standard normal distribution (same as linear version)
# ============================================================
np.random.seed(42)
n_points = 200

gauss_points = np.random.randn(n_points, 2) * 0.7

distances = np.sqrt(gauss_points[:, 0]**2 + gauss_points[:, 1]**2)
dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

# Unit circle reference
n_circle = 120
angles = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
circle_points = np.column_stack([np.cos(angles), np.sin(angles)])

# ============================================================
# Grid lines
# ============================================================
grid_n = 13
grid_range = np.linspace(-2.5, 2.5, grid_n)
grid_res = 80  # higher res for smooth nonlinear curves

h_lines = []
for y_val in grid_range:
    line = np.column_stack([np.linspace(-2.5, 2.5, grid_res), np.full(grid_res, y_val)])
    h_lines.append(line)

v_lines = []
for x_val in grid_range:
    line = np.column_stack([np.full(grid_res, x_val), np.linspace(-2.5, 2.5, grid_res)])
    v_lines.append(line)

all_grid_lines = h_lines + v_lines

# ============================================================
# Define all nonlinear transforms as functions: f(points, t)
# where t in [0,1] interpolates from identity to full transform
# ============================================================

def relu_transform(pts, t):
    """ReLU: max(0, x) — folds negative to zero"""
    return pts * (1 - t) + np.maximum(pts, 0) * t

def sigmoid_transform(pts, t):
    """Sigmoid: 1/(1+exp(-x)) — squashes to (0,1)"""
    target = 1.0 / (1.0 + np.exp(-pts * 3))  # scale input for visibility
    # Map sigmoid output from (0,1) to (-1, 1) range for better visualization
    target = (target - 0.5) * 4  # center and scale
    return pts * (1 - t) + target * t

def tanh_transform(pts, t):
    """Tanh: squashes to (-1,1) — saturates at edges"""
    target = np.tanh(pts * 1.5) * 1.5
    return pts * (1 - t) + target * t

def swirl_transform(pts, t):
    """Swirl/Vortex: rotation angle depends on distance from origin"""
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    theta_offset = t * 2.0 * np.exp(-r * 0.5)  # stronger twist near center
    cos_t = np.cos(theta_offset)
    sin_t = np.sin(theta_offset)
    new_x = pts[:, 0] * cos_t - pts[:, 1] * sin_t
    new_y = pts[:, 0] * sin_t + pts[:, 1] * cos_t
    return np.column_stack([new_x, new_y])

def radial_squeeze(pts, t):
    """Radial Squeeze: compress radius via sqrt, directions preserved"""
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + 1e-10)
    target_r = np.sqrt(r)  # sqrt compresses large, expands small
    scale = (1 - t) * 1.0 + t * (target_r / r)
    return pts * scale[:, np.newaxis]

def polar_warp(pts, t):
    """Polar Warp: maps (x,y) -> (r, theta) as new coordinates"""
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + 1e-10)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    # Scale for visibility: r stays, theta mapped to useful range
    target = np.column_stack([r, theta])
    return pts * (1 - t) + target * t

def sine_ripple(pts, t):
    """Sinusoidal Ripple: add sine wave displacement"""
    dx = t * 0.4 * np.sin(pts[:, 1] * np.pi * 1.5)
    dy = t * 0.4 * np.sin(pts[:, 0] * np.pi * 1.5)
    return pts + np.column_stack([dx, dy])

def softplus_transform(pts, t):
    """Softplus: log(1 + exp(x)) — smooth approximation of ReLU"""
    target = np.log1p(np.exp(pts * 2)) / 2  # scale for visibility
    return pts * (1 - t) + target * t

def exp_mapping(pts, t):
    """Exponential: exponential growth — dramatic nonlinearity"""
    # Use moderate scaling to keep things visible
    target_x = np.sign(pts[:, 0]) * (np.exp(np.abs(pts[:, 0]) * 0.8) - 1)
    target_y = np.sign(pts[:, 1]) * (np.exp(np.abs(pts[:, 1]) * 0.8) - 1)
    target = np.column_stack([target_x, target_y])
    return pts * (1 - t) + target * t

def fisheye_distortion(pts, t):
    """Fisheye / Barrel distortion: inflate center, compress edges"""
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + 1e-10)
    # Barrel: r -> r * (1 + k*r^2)
    k = t * 0.3
    scale = 1.0 + k * r**2
    return pts * scale[:, np.newaxis]

def fold_transform(pts, t):
    """Fold: |x| — folds left half onto right half"""
    target = np.abs(pts)
    return pts * (1 - t) + target * t

def quadratic_transform(pts, t):
    """Quadratic: sign(x)*x^2 — amplifies large values, shrinks small"""
    target = np.sign(pts) * pts**2
    return pts * (1 - t) + target * t


# ============================================================
# Transform definitions
# ============================================================
transforms = [
    {
        "name": "[1/12] ReLU: max(0, x)",
        "func": relu_transform,
        "formula": "f(x,y) = (max(0,x), max(0,y))",
        "desc": "Fold negative region to zero (NN activation)",
        "color": "#FF7043",
    },
    {
        "name": "[2/12] Sigmoid",
        "func": sigmoid_transform,
        "formula": "f(v) = 4*(sigmoid(3v) - 0.5)",
        "desc": "Squash to bounded range, saturate at edges",
        "color": "#AB47BC",
    },
    {
        "name": "[3/12] Tanh",
        "func": tanh_transform,
        "formula": "f(v) = 1.5 * tanh(1.5v)",
        "desc": "Smooth squashing, zero-centered",
        "color": "#5C6BC0",
    },
    {
        "name": "[4/12] Swirl / Vortex",
        "func": swirl_transform,
        "formula": "angle(r) = 2 * exp(-0.5r)",
        "desc": "Rotation depends on distance: twist near center",
        "color": "#26C6DA",
    },
    {
        "name": "[5/12] Radial Squeeze (sqrt)",
        "func": radial_squeeze,
        "formula": "r' = sqrt(r), direction unchanged",
        "desc": "Compress large radius, expand small radius",
        "color": "#66BB6A",
    },
    {
        "name": "[6/12] Polar Warp",
        "func": polar_warp,
        "formula": "f(x,y) = (r, theta)",
        "desc": "Map Cartesian to Polar coordinates",
        "color": "#FFA726",
    },
    {
        "name": "[7/12] Sinusoidal Ripple",
        "func": sine_ripple,
        "formula": "dx=0.4*sin(1.5*pi*y), dy=0.4*sin(1.5*pi*x)",
        "desc": "Wave-like distortion, grid becomes wavy",
        "color": "#EC407A",
    },
    {
        "name": "[8/12] Softplus (smooth ReLU)",
        "func": softplus_transform,
        "formula": "f(v) = log(1+exp(2v)) / 2",
        "desc": "Smooth version of ReLU, no hard fold",
        "color": "#8D6E63",
    },
    {
        "name": "[9/12] Exponential Mapping",
        "func": exp_mapping,
        "formula": "f(v) = sign(v)*(exp(0.8|v|)-1)",
        "desc": "Dramatic expansion at edges, compress center",
        "color": "#EF5350",
    },
    {
        "name": "[10/12] Fisheye / Barrel",
        "func": fisheye_distortion,
        "formula": "r' = r * (1 + 0.3*r^2)",
        "desc": "Inflate center outward, barrel distortion",
        "color": "#42A5F5",
    },
    {
        "name": "[11/12] Fold |x|",
        "func": fold_transform,
        "formula": "f(x,y) = (|x|, |y|)",
        "desc": "Fold all quadrants into first quadrant",
        "color": "#FFCA28",
    },
    {
        "name": "[12/12] Quadratic",
        "func": quadratic_transform,
        "formula": "f(v) = sign(v) * v^2",
        "desc": "Amplify large, shrink small (power law)",
        "color": "#26A69A",
    },
]

n_transforms = len(transforms)

# ============================================================
# Animation parameters
# ============================================================
frames_per_transition = 40
pause_at_identity = 20
pause_at_target = 30
frames_per_transform = pause_at_identity + frames_per_transition + pause_at_target
total_frames = frames_per_transform * n_transforms

FIXED_LIM = 3.0

# ============================================================
# Create Figure
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(9, 9), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(-FIXED_LIM, FIXED_LIM)
ax.set_ylim(-FIXED_LIM, FIXED_LIM)
ax.set_aspect('equal')

# ---- Fixed reference grid (static, gray dotted) ----
for y_val in grid_range:
    ax.plot([-FIXED_LIM, FIXED_LIM], [y_val, y_val],
            color='#333333', linewidth=0.3, linestyle=':', zorder=0)
for x_val in grid_range:
    ax.plot([x_val, x_val], [-FIXED_LIM, FIXED_LIM],
            color='#333333', linewidth=0.3, linestyle=':', zorder=0)

ax.axhline(y=0, color='#555555', linewidth=1.0, zorder=1)
ax.axvline(x=0, color='#555555', linewidth=1.0, zorder=1)

# ---- Fixed unit circle reference (gray dashed) ----
theta_ref = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta_ref), np.sin(theta_ref),
        color='#444444', linewidth=0.8, linestyle='--', zorder=1, alpha=0.5)

# ---- Moving grid lines ----
moving_grid_objs = []
for _ in all_grid_lines:
    line_obj, = ax.plot([], [], color='#FF7043', alpha=0.18, linewidth=0.6, zorder=2)
    moving_grid_objs.append(line_obj)

# ---- Transformed circle outline ----
circle_line, = ax.plot(circle_points[:, 0], circle_points[:, 1],
                       color='#888888', linewidth=1.5, alpha=0.6, zorder=3)

# ---- Gaussian point cloud ----
cmap = plt.cm.plasma
gauss_scatter = ax.scatter(gauss_points[:, 0], gauss_points[:, 1],
                           c=dist_norm, cmap=cmap, s=25, alpha=0.85,
                           edgecolors='white', linewidths=0.2, zorder=5)

# ---- Title and info text ----
title_text = ax.set_title("", fontsize=15, color='white', pad=18, fontweight='bold')

formula_text = ax.text(0.98, 0.98, "", transform=ax.transAxes,
                       fontsize=11, color='#ffd700',
                       fontfamily='monospace', verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                                 edgecolor='#ffd700', alpha=0.85),
                       zorder=10)

desc_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                    fontsize=11, color='#aaaaaa',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                              edgecolor='#555', alpha=0.85),
                    zorder=10)

# "NONLINEAR" badge (always shown, contrasts with linear version)
type_label = ax.text(0.98, 0.02, "NONLINEAR", transform=ax.transAxes,
                     fontsize=11, fontweight='bold', color='#FF7043',
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                               edgecolor='#FF7043', alpha=0.85),
                     zorder=10)

progress_text = ax.text(0.5, 0.98, "", transform=ax.transAxes,
                        fontsize=10, color='#888',
                        horizontalalignment='center',
                        verticalalignment='top',
                        zorder=10)

ax.tick_params(colors='#555555', labelsize=8)
for spine in ax.spines.values():
    spine.set_color('#333333')


# ============================================================
# Animation update
# ============================================================
def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def update(frame):
    transform_idx = frame // frames_per_transform
    frame_in_block = frame % frames_per_transform

    if transform_idx >= n_transforms:
        transform_idx = n_transforms - 1
        frame_in_block = frames_per_transform - 1

    tr = transforms[transform_idx]
    func = tr["func"]

    # Interpolation parameter
    if frame_in_block < pause_at_identity:
        t = 0.0
    elif frame_in_block < pause_at_identity + frames_per_transition:
        raw_t = (frame_in_block - pause_at_identity) / frames_per_transition
        t = smoothstep(raw_t)
    else:
        t = 1.0

    # ---- Update gaussian points ----
    transformed_gauss = func(gauss_points, t)
    gauss_scatter.set_offsets(transformed_gauss)

    # ---- Update circle outline ----
    transformed_circle = func(circle_points, t)
    cx = np.append(transformed_circle[:, 0], transformed_circle[0, 0])
    cy = np.append(transformed_circle[:, 1], transformed_circle[0, 1])
    circle_line.set_data(cx, cy)
    circle_line.set_color(tr["color"])
    circle_line.set_alpha(0.7)

    # ---- Update moving grid ----
    for i, line_obj in enumerate(moving_grid_objs):
        transformed_line = func(all_grid_lines[i], t)
        line_obj.set_data(transformed_line[:, 0], transformed_line[:, 1])
        line_obj.set_color(tr["color"])
        line_obj.set_alpha(0.15)

    # ---- Update text ----
    title_text.set_text(tr["name"])
    title_text.set_color(tr["color"])

    formula_text.set_text(tr["formula"])

    desc_text.set_text(tr["desc"])
    desc_text.set_color(tr["color"])

    type_label.set_color(tr["color"])
    type_label.get_bbox_patch().set_edgecolor(tr["color"])

    # Progress
    dots = ""
    for i in range(n_transforms):
        if i < transform_idx:
            dots += "● "
        elif i == transform_idx:
            dots += "◉ "
        else:
            dots += "○ "
    progress_text.set_text(f"[ {transform_idx+1}/{n_transforms} ] {dots}")

    # Fixed axes
    ax.set_xlim(-FIXED_LIM, FIXED_LIM)
    ax.set_ylim(-FIXED_LIM, FIXED_LIM)
    ax.set_aspect('equal')

    return [gauss_scatter, circle_line, title_text, formula_text, desc_text,
            type_label, progress_text] + moving_grid_objs


# ============================================================
# Generate animation
# ============================================================
print(f"{n_transforms} nonlinear transforms, total frames: {total_frames}")
print("Generating animation (this may take a minute or two)...")

anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

output_path = "/data/workspace/transformer/nonlinear_transforms_animation.gif"
anim.save(output_path, writer='pillow', fps=20, dpi=100)
print(f"Animation saved: {output_path}")
plt.close(fig)


# ============================================================
# Static keyframes (one subplot per transform)
# ============================================================
cols = 4
rows = (n_transforms + cols - 1) // cols
fig2, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), facecolor='#0d1117')
axes = axes.flatten()

for idx in range(len(axes)):
    ax2 = axes[idx]
    ax2.set_facecolor('#0d1117')
    ax2.set_xlim(-FIXED_LIM, FIXED_LIM)
    ax2.set_ylim(-FIXED_LIM, FIXED_LIM)
    ax2.set_aspect('equal')
    ax2.tick_params(colors='#444444', labelsize=6)
    for spine in ax2.spines.values():
        spine.set_color('#333333')

    if idx >= n_transforms:
        ax2.set_visible(False)
        continue

    tr = transforms[idx]
    func = tr["func"]

    # Fixed reference grid
    for y_val in grid_range:
        ax2.plot([-FIXED_LIM, FIXED_LIM], [y_val, y_val],
                 color='#333333', linewidth=0.3, linestyle=':', zorder=0)
    for x_val in grid_range:
        ax2.plot([x_val, x_val], [-FIXED_LIM, FIXED_LIM],
                 color='#333333', linewidth=0.3, linestyle=':', zorder=0)
    ax2.axhline(y=0, color='#555555', linewidth=0.8, zorder=1)
    ax2.axvline(x=0, color='#555555', linewidth=0.8, zorder=1)

    # Fixed unit circle
    ax2.plot(np.cos(theta_ref), np.sin(theta_ref),
             color='#444444', linewidth=0.6, linestyle='--', zorder=1, alpha=0.4)

    # Transformed grid
    for line in all_grid_lines:
        tl = func(line, 1.0)
        ax2.plot(tl[:, 0], tl[:, 1], color=tr["color"], alpha=0.15, linewidth=0.5, zorder=2)

    # Original gaussian (gray ghost)
    ax2.scatter(gauss_points[:, 0], gauss_points[:, 1],
                c='#555555', s=6, alpha=0.2, zorder=3)

    # Transformed gaussian
    tg = func(gauss_points, 1.0)
    ax2.scatter(tg[:, 0], tg[:, 1],
                c=dist_norm, cmap=plt.cm.plasma, s=10, alpha=0.85,
                edgecolors='white', linewidths=0.1, zorder=5)

    # Transformed circle
    tc = func(circle_points, 1.0)
    tcx = np.append(tc[:, 0], tc[0, 0])
    tcy = np.append(tc[:, 1], tc[0, 1])
    ax2.plot(tcx, tcy, color=tr["color"], linewidth=1.2, alpha=0.7, zorder=4)

    ax2.set_title(tr["name"], fontsize=9, color=tr["color"], fontweight='bold', pad=8)
    ax2.text(0.5, -0.06, tr["formula"],
             transform=ax2.transAxes, fontsize=6, color='#ffd700',
             fontfamily='monospace', horizontalalignment='center')

fig2.suptitle("All 12 Nonlinear Transforms in 2D",
              fontsize=18, color='white', fontweight='bold', y=1.01)
plt.tight_layout()

keyframe_path = "/data/workspace/transformer/nonlinear_transforms_keyframes.png"
fig2.savefig(keyframe_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"Keyframes saved: {keyframe_path}")

plt.close('all')
print("Done!")
