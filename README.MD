# ComfyUI Latent Transform Sampler

Advanced latent space transformations during diffusion sampling with precise control over timing and distribution.

## Features

- **Transform Types**: shift, mirror (H/V), rotate (90°/180°), random, sequences, spiral
- **Distribution Strategies**: even, golden ratio, fibonacci, front/back loaded, edges, center, exponential, logarithmic, random, manual
- **Advanced Controls**: transform strength, accumulating shifts, reversible transforms, minimum spacing

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-latent-transform-sampler
```

Restart ComfyUI. Find the node under `sampling/transform > Latent Transform Sampler`

## Requirements

- ComfyUI (latest)
- PyTorch >= 2.0.0
- NumPy >= 1.19.0

## Usage

Connect like a standard KSampler:
- Model, positive/negative conditioning, latent_image inputs
- Output to VAE Decode

### Basic Example

```yaml
transform_count: 3
transform_type: shift
distribution: golden_ratio
shift_pixels_x: 16
shift_pixels_y: 0
```

## Parameters

### Core Settings
- `transform_count` (0-20): Number of transforms to apply
- `transform_type`: Type of transformation
- `distribution`: How to space transforms across steps
- `steps` (1-10000): Total sampling steps

### Transform Settings
- `shift_pixels_x/y` (-512 to 512): Shift amount in pixels
- `transform_strength` (0.0-1.0): Blend strength
- `accumulate_shifts`: Whether shifts accumulate or reset

### Advanced Options
- `min_spacing` (1-10): Minimum steps between transforms
- `reverse_on_second_half`: Undo transforms in second half
- `transform_sequence`: Custom sequence (comma-separated)
- `manual_steps`: Exact steps (comma-separated)
- `debug_mode`: Console output for debugging

## Transform Shorthand

For sequences:
- `s` or `shift` - Shift transform
- `h` or `mh` - Mirror horizontal
- `v` or `mv` - Mirror vertical
- `cw` - Rotate 90° clockwise
- `ccw` - Rotate 90° counter-clockwise
- `180` - Rotate 180°
- `r` - Random

Example: `"s,h,cw,s,v,ccw"`

## License

MIT License - see [LICENSE](LICENSE) file for details.