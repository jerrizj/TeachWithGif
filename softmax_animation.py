"""
Understanding Softmax — A Visual Explanation
=============================================
Two-panel design:
   TOP    = input logits (what goes in)
   BOTTOM = softmax output (what comes out)

This lets us see BOTH the input change AND how the output responds.

Scene Group 1: "The Pipeline"
   logits -> exp() -> normalize
   The exp() step is where the magic happens: small gaps in logits
   become huge gaps after exponentiation (winner-take-all effect).

Scene Group 2: "Temperature"
   Sweep T from high to low.
   High T -> flat / uncertain.  Low T -> sharp / one-hot.
   This is exactly the `temperature` knob in LLM sampling.

Scene Group 3: "Invariance vs Sensitivity"
   Shift  (x + c) -> output does NOT change  (shift-invariant)
   Scale  (2x)    -> output DOES change      (scale-sensitive)
   Note this is the opposite of LayerNorm, which is scale-invariant.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

np.random.seed(7)

# ============================================================
# The logits we will work with
# ============================================================
n_classes = 8
class_labels = [f"c{i}" for i in range(n_classes)]

# Hand-picked logits so the story is clear:
# note c3 (5.5) and c6 (4.5) differ by only 1.0 in logit space,
# but end up 2.7x apart in probability - that's the exp() effect.
logits = np.array([2.0, 4.0, 1.0, 5.5, 3.0, 0.5, 4.5, 2.5])

eps = 1e-12


def softmax(x, T=1.0):
    """Numerically stable softmax with temperature."""
    z = x / T
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + eps)


# --- Pipeline intermediate values ---
exp_vals = np.exp(logits)            # raw exp, before normalization
probs_T1 = softmax(logits, 1.0)      # standard softmax

# --- Temperature sweep values ---
temperatures = [5.0, 2.0, 1.0, 0.5, 0.2]
probs_by_T = {T: softmax(logits, T) for T in temperatures}

# --- Invariance demo ---
SHIFT_C = 5.0
logits_shifted = logits + SHIFT_C
probs_shifted = softmax(logits_shifted, 1.0)   # identical to probs_T1

SCALE_K = 2.0
logits_scaled = logits * SCALE_K
probs_scaled = softmax(logits_scaled, 1.0)     # equals softmax(logits, T=0.5)

zeros = np.zeros(n_classes)

# ============================================================
# Scene timing
# ============================================================
frames_per_scene = 80
pause_frames = 20
transition_frames = frames_per_scene - pause_frames
extra_end_pause = 50   # hold the final state so it does not feel cut off

# ============================================================
# Colors
# ============================================================
PIPE_COLOR = "#4FC3F7"     # pipeline: light blue
INPUT_COLOR = "#78909C"    # top panel default: slate gray

# Temperature colors follow physical intuition: hot = red, cold = blue
temp_colors = {
    5.0: "#EF5350",   # hot   -> flat, uncertain
    2.0: "#FF9800",
    1.0: "#66BB6A",   # standard
    0.5: "#26C6DA",
    0.2: "#42A5F5",   # cold  -> sharp, confident
}

SHIFT_COLOR = "#CE93D8"    # purple
SCALE_COLOR = "#FFB74D"    # orange

# ============================================================
# Build scenes
# Each scene interpolates BOTH panels from *_from to *_to.
# ============================================================
scenes = []


def add_scene(title, subtitle, top_from, top_to, bot_from, bot_to,
              top_ylim_from, top_ylim_to, bot_ylim_from, bot_ylim_to,
              color, badge, bot_label, note=""):
    scenes.append({
        "title": title,
        "subtitle": subtitle,
        "top_from": top_from, "top_to": top_to,
        "bot_from": bot_from, "bot_to": bot_to,
        "top_ylim_from": top_ylim_from, "top_ylim_to": top_ylim_to,
        "bot_ylim_from": bot_ylim_from, "bot_ylim_to": bot_ylim_to,
        "color": color,
        "badge": badge,
        "bot_label": bot_label,
        "note": note,
    })


# ---------- Group 1: the pipeline ----------
add_scene(
    title="Step 1: Input logits (raw scores)",
    subtitle="Arbitrary real numbers - can be any size, positive or negative",
    top_from=zeros, top_to=logits,
    bot_from=zeros, bot_to=zeros,
    top_ylim_from=(0, 6.5), top_ylim_to=(0, 6.5),
    bot_ylim_from=(0, 1.05), bot_ylim_to=(0, 1.05),
    color=PIPE_COLOR, badge="PIPELINE",
    bot_label="Output (empty)",
    note="c3=5.5 is the largest, c6=4.5 is second - gap is only 1.0",
)

add_scene(
    title="Step 2: Apply exp() - this is where e does its work",
    subtitle="Small gaps in logits become HUGE gaps after exponentiation",
    top_from=logits, top_to=logits,
    bot_from=zeros, bot_to=exp_vals,
    top_ylim_from=(0, 6.5), top_ylim_to=(0, 6.5),
    bot_ylim_from=(0, 1.05), bot_ylim_to=(0, 260),
    color=PIPE_COLOR, badge="PIPELINE",
    bot_label="exp(logit)",
    note=f"exp(5.5)={exp_vals[3]:.0f} vs exp(4.5)={exp_vals[6]:.0f} - now 2.7x apart!",
)

add_scene(
    title="Step 3: Normalize - divide by the sum",
    subtitle="Shape stays identical, only the scale changes. Now sums to 1.",
    top_from=logits, top_to=logits,
    bot_from=exp_vals, bot_to=probs_T1,
    top_ylim_from=(0, 6.5), top_ylim_to=(0, 6.5),
    bot_ylim_from=(0, 260), bot_ylim_to=(0, 1.05),
    color=PIPE_COLOR, badge="PIPELINE",
    bot_label="Probability",
    note="Watch the bars stay put while the y-axis shrinks - pure rescaling",
)

# ---------- Group 2: temperature sweep ----------
prev_probs = probs_T1
prev_top = logits
prev_top_ylim = (0, 6.5)

for T in temperatures:
    scaled_logits = logits / T
    top_max = max(6.5, float(np.ceil(scaled_logits.max() * 1.15)))
    cur_top_ylim = (0, top_max)

    if T > 1.0:
        flavour = "HOT -> flatter, more uncertain"
    elif T == 1.0:
        flavour = "Standard softmax (T=1)"
    else:
        flavour = "COLD -> sharper, more confident"

    add_scene(
        title=f"Temperature T = {T}",
        subtitle=f"softmax(logits / {T})   |   {flavour}",
        top_from=prev_top, top_to=scaled_logits,
        bot_from=prev_probs, bot_to=probs_by_T[T],
        top_ylim_from=prev_top_ylim, top_ylim_to=cur_top_ylim,
        bot_ylim_from=(0, 1.05), bot_ylim_to=(0, 1.05),
        color=temp_colors[T], badge=f"TEMPERATURE T={T}",
        bot_label="Probability",
        note=f"max prob = {probs_by_T[T].max():.3f}",
    )
    prev_probs = probs_by_T[T]
    prev_top = scaled_logits
    prev_top_ylim = cur_top_ylim

# ---------- Group 3: invariance vs sensitivity ----------
# Back to baseline T=1 first
add_scene(
    title="Back to baseline: T = 1",
    subtitle="Reset, so we can test two properties next",
    top_from=prev_top, top_to=logits,
    bot_from=prev_probs, bot_to=probs_T1,
    top_ylim_from=prev_top_ylim, top_ylim_to=(0, 6.5),
    bot_ylim_from=(0, 1.05), bot_ylim_to=(0, 1.05),
    color="#66BB6A", badge="BASELINE",
    bot_label="Probability",
    note="This is our reference output",
)

# Shift invariance: add 5 to every logit
add_scene(
    title=f"Test 1: SHIFT - add {SHIFT_C:.0f} to every logit",
    subtitle="Top panel rises... but the bottom panel does NOT move at all!",
    top_from=logits, top_to=logits_shifted,
    bot_from=probs_T1, bot_to=probs_shifted,   # mathematically identical
    top_ylim_from=(0, 6.5), top_ylim_to=(0, 11.5),
    bot_ylim_from=(0, 1.05), bot_ylim_to=(0, 1.05),
    color=SHIFT_COLOR, badge="SHIFT INVARIANT",
    bot_label="Probability (unchanged!)",
    note="softmax(x + c) = softmax(x)  because e^c cancels out",
)

# Scale sensitivity: multiply every logit by 2
add_scene(
    title=f"Test 2: SCALE - multiply every logit by {SCALE_K:.0f}",
    subtitle="Now the output DOES change - it gets sharper!",
    top_from=logits_shifted, top_to=logits_scaled,
    bot_from=probs_shifted, bot_to=probs_scaled,
    top_ylim_from=(0, 11.5), top_ylim_to=(0, 11.5),
    bot_ylim_from=(0, 1.05), bot_ylim_to=(0, 1.05),
    color=SCALE_COLOR, badge="SCALE SENSITIVE",
    bot_label="Probability (sharper!)",
    note=f"softmax(2x) = softmax(x, T=0.5)  ->  max prob {probs_T1.max():.3f} to {probs_scaled.max():.3f}",
)

n_scenes_total = len(scenes)
total_frames = n_scenes_total * frames_per_scene + extra_end_pause

# ============================================================
# Figure: two stacked panels
# ============================================================
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(12, 8), facecolor='#0d1117',
    gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.42}
)

for a in (ax_top, ax_bot):
    a.set_facecolor('#0d1117')
    a.tick_params(colors='#555555', labelsize=8)
    for spine in a.spines.values():
        spine.set_color('#333333')
    a.set_xlim(-0.6, n_classes - 0.4)
    a.set_xticks(range(n_classes))
    a.set_xticklabels(class_labels, color='#777777')

bar_x = np.arange(n_classes)

top_bars = ax_top.bar(bar_x, np.zeros(n_classes), color=INPUT_COLOR,
                      alpha=0.9, edgecolor='white', linewidth=0.4, zorder=3)
bot_bars = ax_bot.bar(bar_x, np.zeros(n_classes), color=PIPE_COLOR,
                      alpha=0.9, edgecolor='white', linewidth=0.4, zorder=3)

ax_top.axhline(y=0, color='#666666', linewidth=1.0, zorder=2)
ax_bot.axhline(y=0, color='#666666', linewidth=1.0, zorder=2)

ax_top.set_ylabel("Input value", fontsize=10, color='#888888')
ax_bot.set_ylabel("Output value", fontsize=10, color='#888888')
ax_bot.set_xlabel("Class", fontsize=10, color='#888888')

# --- Panel headers ---
top_header = ax_top.text(0.01, 1.14, "INPUT  (logits)", transform=ax_top.transAxes,
                         fontsize=10, color='#90A4AE', fontweight='bold',
                         va='bottom', ha='left', zorder=10)
bot_header = ax_bot.text(0.01, 1.05, "OUTPUT", transform=ax_bot.transAxes,
                         fontsize=10, color=PIPE_COLOR, fontweight='bold',
                         va='bottom', ha='left', zorder=10)

# --- Text overlays (laid out to avoid overlap) ---
title_text = fig.text(0.5, 0.965, "", fontsize=15, color='white',
                      fontweight='bold', ha='center', va='top')
subtitle_text = fig.text(0.5, 0.925, "", fontsize=10, color='#aaaaaa',
                         ha='center', va='top')

badge_obj = ax_top.text(0.99, 1.14, "", transform=ax_top.transAxes,
                        fontsize=10, fontweight='bold', color=PIPE_COLOR,
                        va='bottom', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                                  edgecolor=PIPE_COLOR, alpha=0.9),
                        zorder=10)

top_stats = ax_top.text(0.99, 0.94, "", transform=ax_top.transAxes,
                        fontsize=9, color='#B0BEC5', fontfamily='monospace',
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                                  edgecolor='#455A64', alpha=0.85),
                        zorder=10)

bot_stats = ax_bot.text(0.99, 0.94, "", transform=ax_bot.transAxes,
                        fontsize=9, color='#80cbc4', fontfamily='monospace',
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                                  edgecolor='#80cbc4', alpha=0.85),
                        zorder=10)

note_text = fig.text(0.5, 0.048, "", fontsize=9, color='#FFD54F',
                     ha='center', va='bottom', fontfamily='monospace')

progress_text = fig.text(0.5, 0.012, "", fontsize=9, color='#666',
                         ha='center', va='bottom')


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def lerp_pair(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def update(frame):
    scene_idx = frame // frames_per_scene
    frame_in_scene = frame % frames_per_scene

    # Clamp for the extra end-pause frames
    if scene_idx >= n_scenes_total:
        scene_idx = n_scenes_total - 1
        frame_in_scene = frames_per_scene - 1

    sc = scenes[scene_idx]

    t = smoothstep(frame_in_scene / transition_frames) \
        if frame_in_scene < transition_frames else 1.0

    cur_top = sc["top_from"] + (sc["top_to"] - sc["top_from"]) * t
    cur_bot = sc["bot_from"] + (sc["bot_to"] - sc["bot_from"]) * t

    # top panel bars
    for i, bar in enumerate(top_bars):
        bar.set_height(cur_top[i])
    # bottom panel bars, highlight the argmax
    argmax_i = int(np.argmax(cur_bot)) if cur_bot.max() > 0 else -1
    for i, bar in enumerate(bot_bars):
        bar.set_height(cur_bot[i])
        bar.set_color(sc["color"])
        if i == argmax_i:
            bar.set_alpha(1.0)
            bar.set_linewidth(1.4)
            bar.set_edgecolor('white')
        else:
            bar.set_alpha(0.65)
            bar.set_linewidth(0.4)
            bar.set_edgecolor('#cccccc')

    ax_top.set_ylim(*lerp_pair(sc["top_ylim_from"], sc["top_ylim_to"], t))
    ax_bot.set_ylim(*lerp_pair(sc["bot_ylim_from"], sc["bot_ylim_to"], t))

    title_text.set_text(sc["title"])
    title_text.set_color(sc["color"])
    subtitle_text.set_text(sc["subtitle"])
    note_text.set_text(sc["note"])

    badge_obj.set_text(sc["badge"])
    badge_obj.set_color(sc["color"])
    badge_obj.get_bbox_patch().set_edgecolor(sc["color"])

    bot_header.set_text(f'OUTPUT  ({sc["bot_label"]})')
    bot_header.set_color(sc["color"])

    top_stats.set_text(
        f'min={cur_top.min():+.2f}  max={cur_top.max():+.2f}  '
        f'gap={cur_top.max() - np.sort(cur_top)[-2]:+.2f}'
    )
    bot_stats.set_text(
        f'sum={cur_bot.sum():.3f}   max={cur_bot.max():.3f}'
    )

    dots = []
    for i in range(n_scenes_total):
        dots.append("●" if i < scene_idx else ("◉" if i == scene_idx else "○"))
    # groups: pipeline(3) | temperature(5) | invariance(3)
    progress_text.set_text(
        f'[{scene_idx+1}/{n_scenes_total}]   '
        + " ".join(dots[:3]) + "  |  "
        + " ".join(dots[3:8]) + "  |  "
        + " ".join(dots[8:])
    )

    return list(top_bars) + list(bot_bars)


print(f"Softmax explainer: {n_scenes_total} scenes, {total_frames} frames")
print("Generating animation...")

anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

output_path = "/data/workspace/transformer/softmax_animation.gif"
anim.save(output_path, writer='pillow', fps=20, dpi=95)
print(f"Animation saved: {output_path}")
plt.close(fig)

# ============================================================
# Keyframes
#   Row 1 = the pipeline, three panels each with its OWN y-axis
#           so the real numbers are visible (no fake rescaling)
#   Row 2 = temperature sweep | invariance | exp on log scale
# ============================================================
fig2 = plt.figure(figsize=(17, 9.5), facecolor='#0d1117')
gs = fig2.add_gridspec(2, 3, hspace=0.42, wspace=0.34)


def style_ax(a):
    a.set_facecolor('#0d1117')
    a.tick_params(colors='#555555', labelsize=8)
    for spine in a.spines.values():
        spine.set_color('#333333')
    a.set_xticks(range(n_classes))
    a.set_xticklabels(class_labels, color='#777777', fontsize=9)


# ---------- Row 1, panel 1: raw logits ----------
ax1 = fig2.add_subplot(gs[0, 0])
style_ax(ax1)
ax1.set_title("(1) Input logits\nany real numbers",
              fontsize=11, color='#90A4AE', fontweight='bold', pad=8)
ax1.bar(bar_x, logits, color='#78909C', alpha=0.9,
        edgecolor='white', linewidth=0.4)
for i, v in enumerate(logits):
    ax1.text(i, v + 0.15, f"{v:.1f}", ha='center', fontsize=8, color='#B0BEC5')
ax1.set_ylim(0, logits.max() * 1.22)
ax1.set_ylabel("logit value", fontsize=9, color='#888')
ax1.annotate("", xy=(3, logits[3]), xytext=(6, logits[6]),
             arrowprops=dict(arrowstyle='<->', color='#FFD54F', lw=1.1))
ax1.text(4.5, (logits[3] + logits[6]) / 2 + 0.25, "gap = 1.0",
         ha='center', fontsize=8, color='#FFD54F')

# ---------- Row 1, panel 2: after exp() ----------
ax2 = fig2.add_subplot(gs[0, 1])
style_ax(ax2)
ax2.set_title("(2) After exp()\ny-axis = exp(logit), gap EXPLODES",
              fontsize=11, color='#FF7043', fontweight='bold', pad=8)
ax2.bar(bar_x, exp_vals, color='#FF7043', alpha=0.9,
        edgecolor='white', linewidth=0.4)
for i, v in enumerate(exp_vals):
    ax2.text(i, v + 8, f"{v:.0f}", ha='center', fontsize=8, color='#FFCC80')
ax2.set_ylim(0, exp_vals.max() * 1.22)
ax2.annotate("", xy=(3, exp_vals[3]), xytext=(6, exp_vals[6]),
             arrowprops=dict(arrowstyle='<->', color='#FFD54F', lw=1.1))
ax2.text(4.5, (exp_vals[3] + exp_vals[6]) / 2 + 10,
         f"gap = {exp_vals[3] - exp_vals[6]:.0f}\n({exp_vals[3]/exp_vals[6]:.1f}x ratio)",
         ha='center', fontsize=8, color='#FFD54F')

# ---------- Row 1, panel 3: after normalize ----------
ax3 = fig2.add_subplot(gs[0, 2])
style_ax(ax3)
ax3.set_title(f"(3) Divide by sum = {exp_vals.sum():.0f}\ny-axis = probability, sums to 1",
              fontsize=11, color=PIPE_COLOR, fontweight='bold', pad=8)
ax3.bar(bar_x, probs_T1, color=PIPE_COLOR, alpha=0.9,
        edgecolor='white', linewidth=0.4)
for i, v in enumerate(probs_T1):
    ax3.text(i, v + 0.018, f"{v:.3f}", ha='center', fontsize=8, color='#B3E5FC')
ax3.set_ylim(0, probs_T1.max() * 1.25)
ax3.text(0.99, 0.97, f"sum = {probs_T1.sum():.3f}",
         transform=ax3.transAxes, fontsize=9, color=PIPE_COLOR,
         ha='right', va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                   edgecolor=PIPE_COLOR, alpha=0.9))

# --- arrows between row-1 panels, placed in the real blank gaps ---
# Resolve the actual axes positions first so nothing can overlap.
fig2.canvas.draw()
_p1 = ax1.get_position()
_p2 = ax2.get_position()
_p3 = ax3.get_position()
_y_mid = (_p1.y0 + _p1.y1) / 2
_gap_a = (_p1.x1 + _p2.x0) / 2      # blank gap between panel 1 and 2
_gap_b = (_p2.x1 + _p3.x0) / 2      # blank gap between panel 2 and 3

for _gx, _label, _col in ((_gap_a, "exp()", '#FF7043'),
                          (_gap_b, "/ sum", PIPE_COLOR)):
    fig2.text(_gx, _y_mid + 0.012, _label, fontsize=11, color=_col,
              fontweight='bold', ha='center', va='bottom')
    fig2.text(_gx, _y_mid - 0.004, "\u25b6", fontsize=14, color=_col,
              ha='center', va='top')

# ---------- Row 2, panel 1: temperature sweep ----------
ax4 = fig2.add_subplot(gs[1, 0])
style_ax(ax4)
ax4.set_title("Temperature: flat (hot) -> sharp (cold)",
              fontsize=11, color='#66BB6A', fontweight='bold', pad=8)
bw = 0.16
for k, T in enumerate(temperatures):
    ax4.bar(bar_x + (k - 2) * bw, probs_by_T[T], width=bw,
            color=temp_colors[T], alpha=0.9,
            label=f"T={T}  max={probs_by_T[T].max():.3f}")
ax4.set_ylabel("probability", fontsize=9, color='#888')
ax4.set_xlabel("class", fontsize=9, color='#888')
ax4.legend(fontsize=7.5, loc='upper left', framealpha=0.6,
           facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

# ---------- Row 2, panel 2: invariance ----------
ax5 = fig2.add_subplot(gs[1, 1])
style_ax(ax5)
ax5.set_title("Shift invariant, but scale sensitive",
              fontsize=11, color=SHIFT_COLOR, fontweight='bold', pad=8)
w2 = 0.27
ax5.bar(bar_x - w2, probs_T1, width=w2, color='#66BB6A',
        alpha=0.9, label=f'softmax(x)        max={probs_T1.max():.3f}')
ax5.bar(bar_x, probs_shifted, width=w2, color=SHIFT_COLOR,
        alpha=0.9, label=f'softmax(x+5)   max={probs_shifted.max():.3f}  SAME')
ax5.bar(bar_x + w2, probs_scaled, width=w2, color=SCALE_COLOR,
        alpha=0.9, label=f'softmax(2x)     max={probs_scaled.max():.3f}  sharper')
ax5.set_ylabel("probability", fontsize=9, color='#888')
ax5.set_xlabel("class", fontsize=9, color='#888')
ax5.legend(fontsize=7.5, loc='upper left', framealpha=0.6,
           facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

# ---------- Row 2, panel 3: exp on log scale ----------
ax6 = fig2.add_subplot(gs[1, 2])
style_ax(ax6)
ax6.set_title("Same exp() values on a LOG axis\n(small ones become visible)",
              fontsize=11, color='#FFD54F', fontweight='bold', pad=8)
ax6.bar(bar_x, exp_vals, color='#FF7043', alpha=0.9,
        edgecolor='white', linewidth=0.4)
ax6.set_yscale('log')
ax6.set_ylim(1, exp_vals.max() * 6)
ax6.set_ylabel("exp(logit)  [log scale]", fontsize=9, color='#888')
ax6.set_xlabel("class", fontsize=9, color='#888')
for i, v in enumerate(exp_vals):
    ax6.text(i, v * 1.25, f"{v:.0f}", ha='center', fontsize=8, color='#FFD54F')

fig2.suptitle("Understanding Softmax - Visual Explanation",
              fontsize=17, color='white', fontweight='bold', y=0.975)

keyframe_path = "/data/workspace/transformer/softmax_keyframes.png"
fig2.savefig(keyframe_path, dpi=140, bbox_inches='tight', facecolor='#0d1117')
print(f"Keyframes saved: {keyframe_path}")

plt.close('all')
print("Done!")
