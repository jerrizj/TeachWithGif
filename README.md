# Deep Learning Visualizations & Experiments

A collection of deep learning experiments, Transformer implementations, and **interactive geometric visualizations** that reveal how neural networks transform space.

## Highlights: Space Transform Animations

The core of this project is a set of animated visualizations showing how **linear** and **nonlinear** transformations act on 2D point clouds — the fundamental building blocks of neural networks.

### Linear Transforms (`transformer/linear_transforms_animation.py`)

Demonstrates **12 types of linear transforms** applied to a 2D Gaussian-distributed point cloud, with a fixed coordinate system for reference:

| # | Transform | Key Property |
|---|-----------|-------------|
| 1 | Uniform Scaling | Equal stretch in all directions |
| 2 | Non-uniform Scaling | Stretch X, compress Y — circle becomes ellipse |
| 3 | Rotation (45°) | Preserves distances and angles |
| 4 | Horizontal Shear | X shifts proportionally to Y |
| 5 | Vertical Shear | Y shifts proportionally to X |
| 6 | Reflection (X-axis) | Vertical flip |
| 7 | Reflection (Y-axis) | Horizontal flip |
| 8 | Reflection (Origin) | Equivalent to 180° rotation |
| 9 | Reflection (y=x) | Swap coordinates (transpose) |
| 10 | Projection (X-axis) | Collapse to 1D — singular, irreversible |
| 11 | Projection (Y-axis) | Collapse to 1D — singular, irreversible |
| 12 | Rotation + Scaling | Composite transform |

![Linear Transforms Keyframes](transformer/linear_transforms_keyframes.png)

**Key observation:** Under linear transforms, straight grid lines always remain straight and parallel. The unit circle may deform into an ellipse but never into a more complex shape.

---

### Nonlinear Transforms (`transformer/nonlinear_transforms_animation.py`)

Demonstrates **12 types of nonlinear transforms** — the kind of operations that give neural networks their expressive power:

| # | Transform | Key Property |
|---|-----------|-------------|
| 1 | ReLU `max(0,x)` | Hard fold — clips negative region to zero |
| 2 | Sigmoid | Smooth squash into bounded range, saturates at edges |
| 3 | Tanh | Zero-centered smooth squash |
| 4 | Swirl / Vortex | Rotation angle depends on distance from origin |
| 5 | Radial Squeeze `√r` | Compress large radii, expand small radii |
| 6 | Polar Warp | Map Cartesian → Polar coordinates |
| 7 | Sinusoidal Ripple | Wave-like distortion, grid becomes wavy |
| 8 | Softplus | Smooth approximation of ReLU |
| 9 | Exponential | Dramatic expansion at edges |
| 10 | Fisheye / Barrel | Center inflates outward |
| 11 | Fold `|x|` | All quadrants fold into first quadrant |
| 12 | Quadratic | Power-law amplification |

![Nonlinear Transforms Keyframes](transformer/nonlinear_transforms_keyframes.png)

**Key observation:** Unlike linear transforms, nonlinear transforms **bend and curve** the grid lines. This is exactly why neural networks need nonlinear activation functions — only nonlinearity can learn complex decision boundaries.

---

### Neural Network Transform (`transformer/nn_transform_animation.py`)

Combines both: shows a full **neural network forward pass** (Linear → ReLU → Linear → ReLU) solving the classic **XOR problem** — a dataset that is linearly inseparable in the original space but becomes separable after the network transforms the geometry step by step.

![NN Transform Keyframes](transformer/nn_transform_keyframes.png)

---

## Other Experiments

| File | Description |
|------|-------------|
| `transformer/llm.py` | Large Language Model implementation |
| `transformer/Transformer-from-scratch/` | Transformer architecture built from scratch |
| `testMNIST.py` | MNIST handwritten digit classification |
| `testMNISTOnRay.py` | Distributed MNIST training using Ray |
| `tensorRt.py` | TensorRT inference optimization |
| `show_mnist.py` | MNIST sample visualization |

## Quick Start

```bash
# Generate linear transforms animation
python transformer/linear_transforms_animation.py

# Generate nonlinear transforms animation
python transformer/nonlinear_transforms_animation.py

# Generate neural network space transform animation
python transformer/nn_transform_animation.py
```

### Requirements

- Python 3.8+
- NumPy
- Matplotlib (with Pillow for GIF export)

## Why This Matters

Understanding how neural networks transform space geometrically is one of the most powerful intuitions in deep learning:

1. **Linear layers** rotate, scale, shear, and project — powerful but limited to keeping straight lines straight
2. **Nonlinear activations** bend and fold space — enabling the network to carve complex decision boundaries
3. **Stacking layers** (linear + nonlinear + linear + ...) progressively untangles the data until it becomes linearly separable

These animations make that abstract concept visually concrete.
