"""
2D Linear Transforms Animation
===============================
展示二维空间中尽可能多的线性变换类型，每种变换都作用在同一组点和网格上，
坐标系（坐标轴范围）始终固定不变，只有点和网格在动。

包含的线性变换：
  1. 均匀缩放 (Uniform Scaling)
  2. 非均匀缩放 (Non-uniform Scaling)
  3. 旋转 (Rotation)
  4. 水平剪切 (Horizontal Shear)
  5. 垂直剪切 (Vertical Shear)
  6. X轴反射 (Reflection about X-axis)
  7. Y轴反射 (Reflection about Y-axis)
  8. 原点反射 (Reflection about Origin)
  9. y=x 反射 (Reflection about y=x)
  10. 向X轴投影 (Projection onto X-axis)
  11. 向Y轴投影 (Projection onto Y-axis)
  12. 旋转+缩放 复合 (Rotation + Scaling)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

# ============================================================
# 生成展示用的点集 —— 二维标准正态分布
# ============================================================
np.random.seed(42)
n_points = 200

# 主体点云：标准正态分布 N(0, I)，均值为原点，标准差为 0.7
gauss_points = np.random.randn(n_points, 2) * 0.7

# 按照到原点的距离给点上色（形成从内到外的渐变，方便观察形变）
distances = np.sqrt(gauss_points[:, 0]**2 + gauss_points[:, 1]**2)
# 归一化到 [0, 1]
dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

# 生成单位圆上的等距点（辅助参考，方便观察圆->椭圆等形变）
n_circle = 80
angles = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
circle_points = np.column_stack([np.cos(angles), np.sin(angles)])

# ============================================================
# 网格线（固定参考坐标系 + 随变换运动的网格）
# ============================================================
grid_n = 11
grid_range = np.linspace(-2.0, 2.0, grid_n)
grid_res = 80

h_lines = []
for y_val in grid_range:
    line = np.column_stack([np.linspace(-2.0, 2.0, grid_res), np.full(grid_res, y_val)])
    h_lines.append(line)

v_lines = []
for x_val in grid_range:
    line = np.column_stack([np.full(grid_res, x_val), np.linspace(-2.0, 2.0, grid_res)])
    v_lines.append(line)

all_grid_lines = h_lines + v_lines

# ============================================================
# 定义所有线性变换
# ============================================================
transforms = [
    {
        "name": "[1/12] Uniform Scaling",
        "matrix": np.array([[1.8, 0.0],
                            [0.0, 1.8]]),
        "formula": "A = [[1.8, 0], [0, 1.8]]",
        "desc": "Scale equally in all directions",
        "color": "#4FC3F7",
    },
    {
        "name": "[2/12] Non-uniform Scaling",
        "matrix": np.array([[1.8, 0.0],
                            [0.0, 0.5]]),
        "formula": "A = [[1.8, 0], [0, 0.5]]",
        "desc": "Stretch X, compress Y",
        "color": "#81C784",
    },
    {
        "name": "[3/12] Rotation (45 deg)",
        "matrix": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                            [np.sin(np.pi/4),  np.cos(np.pi/4)]]),
        "formula": "A = [[cos45, -sin45], [sin45, cos45]]",
        "desc": "CCW 45 deg, preserves distances & angles",
        "color": "#FFB74D",
    },
    {
        "name": "[4/12] Horizontal Shear",
        "matrix": np.array([[1.0, 1.0],
                            [0.0, 1.0]]),
        "formula": "A = [[1, 1], [0, 1]]",
        "desc": "X shifts by Y amount, Y unchanged",
        "color": "#CE93D8",
    },
    {
        "name": "[5/12] Vertical Shear",
        "matrix": np.array([[1.0, 0.0],
                            [0.8, 1.0]]),
        "formula": "A = [[1, 0], [0.8, 1]]",
        "desc": "Y shifts by X amount, X unchanged",
        "color": "#F48FB1",
    },
    {
        "name": "[6/12] Reflect about X-axis",
        "matrix": np.array([[1.0,  0.0],
                            [0.0, -1.0]]),
        "formula": "A = [[1, 0], [0, -1]]",
        "desc": "Flip vertically, X unchanged",
        "color": "#EF5350",
    },
    {
        "name": "[7/12] Reflect about Y-axis",
        "matrix": np.array([[-1.0, 0.0],
                            [ 0.0, 1.0]]),
        "formula": "A = [[-1, 0], [0, 1]]",
        "desc": "Flip horizontally, Y unchanged",
        "color": "#AB47BC",
    },
    {
        "name": "[8/12] Reflect about Origin",
        "matrix": np.array([[-1.0,  0.0],
                            [ 0.0, -1.0]]),
        "formula": "A = [[-1, 0], [0, -1]]",
        "desc": "Same as 180 deg rotation",
        "color": "#7E57C2",
    },
    {
        "name": "[9/12] Reflect about y=x",
        "matrix": np.array([[0.0, 1.0],
                            [1.0, 0.0]]),
        "formula": "A = [[0, 1], [1, 0]]",
        "desc": "Swap x and y (transpose)",
        "color": "#26A69A",
    },
    {
        "name": "[10/12] Project onto X-axis",
        "matrix": np.array([[1.0, 0.0],
                            [0.0, 0.0]]),
        "formula": "A = [[1, 0], [0, 0]]",
        "desc": "Drop Y component, collapse to X-axis",
        "color": "#FFA726",
    },
    {
        "name": "[11/12] Project onto Y-axis",
        "matrix": np.array([[0.0, 0.0],
                            [0.0, 1.0]]),
        "formula": "A = [[0, 0], [0, 1]]",
        "desc": "Drop X component, collapse to Y-axis",
        "color": "#FF7043",
    },
    {
        "name": "[12/12] Rotation + Scaling (composite)",
        "matrix": 1.5 * np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                                   [np.sin(np.pi/6),  np.cos(np.pi/6)]]),
        "formula": "A = 1.5 * R(30 deg)",
        "desc": "Rotate 30 deg then scale 1.5x",
        "color": "#42A5F5",
    },
]

n_transforms = len(transforms)

# ============================================================
# 动画参数
# ============================================================
frames_per_transition = 40    # 从恒等到目标变换的帧数
pause_at_identity = 20        # 在恒等矩阵处暂停的帧数
pause_at_target = 30          # 在目标变换处暂停的帧数
frames_per_transform = pause_at_identity + frames_per_transition + pause_at_target
total_frames = frames_per_transform * n_transforms

# 固定坐标范围
FIXED_LIM = 3.0

# ============================================================
# 创建 Figure
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(9, 9), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(-FIXED_LIM, FIXED_LIM)
ax.set_ylim(-FIXED_LIM, FIXED_LIM)
ax.set_aspect('equal')

# ---- 固定的参考坐标系网格（浅灰虚线，始终不动）----
for y_val in grid_range:
    ax.plot([-FIXED_LIM, FIXED_LIM], [y_val, y_val],
            color='#333333', linewidth=0.3, linestyle=':', zorder=0)
for x_val in grid_range:
    ax.plot([x_val, x_val], [-FIXED_LIM, FIXED_LIM],
            color='#333333', linewidth=0.3, linestyle=':', zorder=0)

# 固定坐标轴（深色十字线）
ax.axhline(y=0, color='#555555', linewidth=1.0, zorder=1)
ax.axvline(x=0, color='#555555', linewidth=1.0, zorder=1)

# ---- 单位圆参考线（灰色虚线圆，固定不动）----
theta_ref = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta_ref), np.sin(theta_ref),
        color='#444444', linewidth=0.8, linestyle='--', zorder=1, alpha=0.5)

# ---- 运动的网格线（随变换一起动）----
moving_grid_objs = []
for _ in all_grid_lines:
    line_obj, = ax.plot([], [], color='#4FC3F7', alpha=0.15, linewidth=0.6, zorder=2)
    moving_grid_objs.append(line_obj)

# ---- 变换后的圆轮廓线（用线而非散点，更清晰）----
circle_line, = ax.plot(circle_points[:, 0], circle_points[:, 1],
                       color='#888888', linewidth=1.2, alpha=0.5, zorder=3)

# ---- 正态分布点云 ----
# 使用 colormap 按距离上色：中心亮，边缘暗
cmap = plt.cm.plasma
gauss_scatter = ax.scatter(gauss_points[:, 0], gauss_points[:, 1],
                           c=dist_norm, cmap=cmap, s=25, alpha=0.85,
                           edgecolors='white', linewidths=0.2, zorder=5)

# ---- 基向量箭头 (e1, e2) 的原始位置（灰色虚箭头，固定不动）----
ax.annotate('', xy=(1, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5, linestyle='--'),
            zorder=3)
ax.annotate('', xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5, linestyle='--'),
            zorder=3)
ax.text(1.05, -0.15, 'e₁', color='#666666', fontsize=10, zorder=3)
ax.text(-0.2, 1.1, 'e₂', color='#666666', fontsize=10, zorder=3)

# ---- 变换后的基向量箭头（彩色，动态更新）----
arrow_e1 = ax.annotate('', xy=(1, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2.5),
                        zorder=6)
arrow_e2 = ax.annotate('', xy=(0, 1), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#69f0ae', lw=2.5),
                        zorder=6)

# 基向量标签
e1_label = ax.text(1.05, 0.1, "Ae₁", color='#ff6b6b', fontsize=11,
                   fontweight='bold', zorder=7)
e2_label = ax.text(0.1, 1.05, "Ae₂", color='#69f0ae', fontsize=11,
                   fontweight='bold', zorder=7)

# ---- 标题和信息文字 ----
title_text = ax.set_title("", fontsize=15, color='white', pad=18, fontweight='bold')

# 矩阵显示
matrix_text = ax.text(0.98, 0.98, "", transform=ax.transAxes,
                      fontsize=12, color='#ffd700',
                      fontfamily='monospace', verticalalignment='top',
                      horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                                edgecolor='#ffd700', alpha=0.85),
                      zorder=10)

# 描述文字
desc_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                    fontsize=11, color='#aaaaaa',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                              edgecolor='#555', alpha=0.85),
                    zorder=10)

# 行列式信息
det_text = ax.text(0.98, 0.02, "", transform=ax.transAxes,
                   fontsize=11, color='#ce93d8',
                   fontfamily='monospace',
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                             edgecolor='#ce93d8', alpha=0.85),
                   zorder=10)

# 进度指示
progress_text = ax.text(0.5, 0.98, "", transform=ax.transAxes,
                        fontsize=10, color='#888',
                        horizontalalignment='center',
                        verticalalignment='top',
                        zorder=10)

# 坐标轴刻度样式
ax.tick_params(colors='#555555', labelsize=8)
for spine in ax.spines.values():
    spine.set_color('#333333')


# ============================================================
# 动画更新函数
# ============================================================
def smoothstep(t):
    """平滑插值 easing"""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def update(frame):
    # 确定当前是哪个变换、以及插值进度
    transform_idx = frame // frames_per_transform
    frame_in_block = frame % frames_per_transform

    if transform_idx >= n_transforms:
        transform_idx = n_transforms - 1
        frame_in_block = frames_per_transform - 1

    tr = transforms[transform_idx]
    target_matrix = tr["matrix"]
    identity = np.eye(2)

    # 计算插值参数 t: 0=恒等, 1=目标变换
    if frame_in_block < pause_at_identity:
        t = 0.0
    elif frame_in_block < pause_at_identity + frames_per_transition:
        raw_t = (frame_in_block - pause_at_identity) / frames_per_transition
        t = smoothstep(raw_t)
    else:
        t = 1.0

    # 当前变换矩阵 = 插值(I, A)
    current_matrix = identity + t * (target_matrix - identity)

    # ---- 更新正态分布点云 ----
    transformed_gauss = gauss_points @ current_matrix.T
    gauss_scatter.set_offsets(transformed_gauss)

    # ---- 更新圆轮廓线 ----
    transformed_circle = circle_points @ current_matrix.T
    # 闭合圆
    cx = np.append(transformed_circle[:, 0], transformed_circle[0, 0])
    cy = np.append(transformed_circle[:, 1], transformed_circle[0, 1])
    circle_line.set_data(cx, cy)
    circle_line.set_color(tr["color"])
    circle_line.set_alpha(0.6)

    # ---- 更新运动网格 ----
    for i, line_obj in enumerate(moving_grid_objs):
        transformed_line = all_grid_lines[i] @ current_matrix.T
        line_obj.set_data(transformed_line[:, 0], transformed_line[:, 1])
        line_obj.set_color(tr["color"])
        line_obj.set_alpha(0.12)

    # ---- 更新基向量箭头 ----
    new_e1 = current_matrix @ np.array([1.0, 0.0])
    new_e2 = current_matrix @ np.array([0.0, 1.0])

    arrow_e1.xy = (new_e1[0], new_e1[1])
    arrow_e2.xy = (new_e2[0], new_e2[1])

    e1_label.set_position((new_e1[0] + 0.08, new_e1[1] + 0.08))
    e2_label.set_position((new_e2[0] + 0.08, new_e2[1] + 0.08))

    # ---- 更新文字信息 ----
    title_text.set_text(tr["name"])
    title_text.set_color(tr["color"])

    m = target_matrix
    matrix_str = f'{tr["formula"]}\n[[{m[0,0]:+.2f}, {m[0,1]:+.2f}]\n [{m[1,0]:+.2f}, {m[1,1]:+.2f}]]'
    matrix_text.set_text(matrix_str)

    desc_text.set_text(tr["desc"])
    desc_text.set_color(tr["color"])

    det_val = np.linalg.det(target_matrix)
    det_text.set_text(f"det(A) = {det_val:+.2f}")

    # 进度条
    dots = ""
    for i in range(n_transforms):
        if i < transform_idx:
            dots += "● "
        elif i == transform_idx:
            dots += "◉ "
        else:
            dots += "○ "
    progress_text.set_text(f"[ {transform_idx+1}/{n_transforms} ] {dots}")

    # 固定坐标轴范围（核心！坐标系不动）
    ax.set_xlim(-FIXED_LIM, FIXED_LIM)
    ax.set_ylim(-FIXED_LIM, FIXED_LIM)
    ax.set_aspect('equal')

    return [gauss_scatter, circle_line, title_text, matrix_text, desc_text,
            det_text, progress_text, arrow_e1, arrow_e2,
            e1_label, e2_label] + moving_grid_objs


# ============================================================
# 生成动画
# ============================================================
print(f"{n_transforms} linear transforms, total frames: {total_frames}")
print("Generating animation (this may take a minute or two)...")

anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

output_path = "/data/workspace/transformer/linear_transforms_animation.gif"
anim.save(output_path, writer='pillow', fps=20, dpi=100)
print(f"Animation saved: {output_path}")
plt.close(fig)


# ============================================================
# 同时保存关键帧静态图 (每种变换一个小图)
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
    M = tr["matrix"]

    # 固定参考网格
    for y_val in grid_range:
        ax2.plot([-FIXED_LIM, FIXED_LIM], [y_val, y_val],
                 color='#333333', linewidth=0.3, linestyle=':', zorder=0)
    for x_val in grid_range:
        ax2.plot([x_val, x_val], [-FIXED_LIM, FIXED_LIM],
                 color='#333333', linewidth=0.3, linestyle=':', zorder=0)
    ax2.axhline(y=0, color='#555555', linewidth=0.8, zorder=1)
    ax2.axvline(x=0, color='#555555', linewidth=0.8, zorder=1)

    # 固定的单位圆参考（灰色虚线）
    ax2.plot(np.cos(theta_ref), np.sin(theta_ref),
             color='#444444', linewidth=0.6, linestyle='--', zorder=1, alpha=0.4)

    # 变换后的网格
    for line in all_grid_lines:
        tl = line @ M.T
        ax2.plot(tl[:, 0], tl[:, 1], color=tr["color"], alpha=0.15, linewidth=0.5, zorder=2)

    # 原始正态分布点（灰色半透明）
    ax2.scatter(gauss_points[:, 0], gauss_points[:, 1],
                c='#555555', s=6, alpha=0.2, zorder=3)

    # 变换后的正态分布点
    tg = gauss_points @ M.T
    ax2.scatter(tg[:, 0], tg[:, 1],
                c=dist_norm, cmap=plt.cm.plasma, s=10, alpha=0.85,
                edgecolors='white', linewidths=0.1, zorder=5)

    # 变换后的圆轮廓线
    tc = circle_points @ M.T
    tcx = np.append(tc[:, 0], tc[0, 0])
    tcy = np.append(tc[:, 1], tc[0, 1])
    ax2.plot(tcx, tcy, color=tr["color"], linewidth=1.0, alpha=0.6, zorder=4)

    # 原始基向量（灰色虚线）
    ax2.annotate('', xy=(1, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=1, ls='--'), zorder=3)
    ax2.annotate('', xy=(0, 1), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=1, ls='--'), zorder=3)

    # 变换后基向量
    new_e1 = M @ np.array([1.0, 0.0])
    new_e2 = M @ np.array([0.0, 1.0])
    ax2.annotate('', xy=new_e1, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2), zorder=6)
    ax2.annotate('', xy=new_e2, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#69f0ae', lw=2), zorder=6)

    det_val = np.linalg.det(M)
    ax2.set_title(tr["name"], fontsize=9, color=tr["color"], fontweight='bold', pad=8)
    ax2.text(0.5, -0.06, f'{tr["formula"]}   det={det_val:+.2f}',
             transform=ax2.transAxes, fontsize=7, color='#ffd700',
             fontfamily='monospace', horizontalalignment='center')

fig2.suptitle("All 12 Linear Transforms in 2D",
              fontsize=18, color='white', fontweight='bold', y=1.01)
plt.tight_layout()

keyframe_path = "/data/workspace/transformer/linear_transforms_keyframes.png"
fig2.savefig(keyframe_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"Keyframes saved: {keyframe_path}")

plt.close('all')
print("Done!")
