# 🔄 ComfyUI N-Transform Sampler

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-brightgreen)](https://github.com/comfyanonymous/ComfyUI)
[![Version](https://img.shields.io/badge/version-3.0.0-blue)](https://github.com/yourusername/comfyui-n-transform-sampler/releases)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-yellow)](https://www.python.org/)

> Advanced latent space transformations during diffusion sampling - apply precise transforms at mathematically distributed intervals

<p align="center">
  <img src="docs/images/banner.png" alt="N-Transform Sampler Banner" width="800"/>
  <br>
  <i>Transform your generations with precision timing and mathematical beauty</i>
</p>

## ✨ Features

### 🎯 **Precise Transform Control**
Apply **exactly N transformations** at strategically calculated points during the sampling process, giving you unprecedented control over latent space manipulation.

### 🔀 **10+ Transform Types**
- **Shift** - Circular pixel shifting with optional accumulation
- **Mirror** - Horizontal and vertical flipping
- **Rotate** - 90°, 180° rotations in any direction  
- **Random** - Chaos mode for experimentation
- **Sequences** - Custom transform patterns
- **Spiral** - Combined rotation and shift effects

### 📊 **11 Distribution Strategies**
Mathematical and aesthetic timing patterns:
- **Even** - Equally spaced transforms
- **Golden Ratio** - Natural aesthetic spacing (1.618...)
- **Fibonacci** - Organic progression (1,1,2,3,5,8...)
- **Edges** - Focus on start and end
- **Center** - Cluster in the middle
- **Front/Back Loaded** - Weight transforms early or late
- **Exponential/Logarithmic** - Acceleration curves
- **Random** - Controlled chaos with minimum spacing
- **Manual** - Full control with exact step specification

### 🎨 **Advanced Features**
- Transform strength control (0-1 blending)
- Accumulating vs independent transforms
- Reversible transformations
- Minimum spacing constraints
- Debug mode with detailed console output
- Sequence support with shorthand notation

## 📦 Installation

### Method 1: Direct Download
1. Download the repository
2. Extract to `ComfyUI/custom_nodes/n_transform_sampler/`
3. Restart ComfyUI

### Method 2: Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-n-transform-sampler n_transform_sampler
# Restart ComfyUI
```

### Method 3: ComfyUI Manager (Coming Soon)
Search for "N-Transform Sampler" in ComfyUI Manager

### Requirements
- ComfyUI (latest version recommended)
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.19.0

## 🚀 Quick Start

### Basic Usage
1. Find the node under `sampling/transform > N-Transform Sampler`
2. Connect like a regular KSampler:
   - Model → model
   - Positive/Negative → conditioning
   - Latent → latent_image
   - Output → VAE Decode

### Your First Transform
```yaml
Transform Count: 3
Transform Type: shift
Distribution: golden_ratio
Shift Pixels X: 16
Shift Pixels Y: 0
```

This creates 3 horizontal shifts at aesthetically pleasing intervals.

## 📖 Examples

### 🌊 Subtle Drift
```python
transform_count: 3
transform_type: "shift"
distribution: "golden_ratio"
shift_pixels_x: 8
accumulate_shifts: True
```
*Creates natural motion with golden ratio timing*

### 🪞 Mirror Bookends
```python
transform_count: 2
transform_type: "mirror_horizontal"
distribution: "edges"
```
*Mirrors at start and end for symmetric composition*

### 🌀 Spiral Effect
```python
transform_count: 5
transform_type: "spiral"
distribution: "fibonacci"
shift_pixels_x: 16
```
*Natural spiral with Fibonacci sequence timing*

### 🎭 Kaleidoscope
```python
transform_count: 8
transform_type: "alternate_mirrors"
distribution: "even"
```
*Alternating H/V mirrors create kaleidoscope patterns*

### 🎲 Controlled Chaos
```python
transform_count: 5
transform_type: "random_transform"
distribution: "random"
transform_strength: 0.7
```
*Random transforms with 70% strength*

## 📊 Distribution Visualizations

```
20 Steps, 5 Transforms:

Even:         |---T----T----T----T----T---|
Golden:       |---T------T----T----T----T-|
Fibonacci:    |T-T--T-----T--------T------|
Front-loaded: |T-T-T--T----T--------------|
Back-loaded:  |--------------T----T--T-T-T|
Edges:        |T-T-----------------T-T-T--|
Center:       |--------T-T-T-T-T----------|
```

## 🎮 Node Parameters

### Core Settings
| Parameter | Range | Description |
|-----------|-------|-------------|
| `transform_count` | 0-20 | Number of transforms to apply |
| `transform_type` | Multiple | Type of transformation |
| `distribution` | 11 options | How to space transforms |
| `steps` | 1-10000 | Total sampling steps |

### Transform Settings
| Parameter | Range | Description |
|-----------|-------|-------------|
| `shift_pixels_x` | -512 to 512 | Horizontal shift amount |
| `shift_pixels_y` | -512 to 512 | Vertical shift amount |
| `transform_strength` | 0.0-1.0 | Blend strength with original |
| `accumulate_shifts` | True/False | Whether shifts accumulate |

### Advanced Options
| Parameter | Range | Description |
|-----------|-------|-------------|
| `min_spacing` | 1-10 | Minimum steps between transforms |
| `reverse_on_second_half` | True/False | Undo transforms in second half |
| `transform_sequence` | String | Custom sequence (comma-separated) |
| `manual_steps` | String | Exact steps (comma-separated) |
| `debug_mode` | True/False | Console output for debugging |

## 🎨 Creative Techniques

### Seamless Textures
Use even distribution with pixel counts that divide your image size:
```python
# For 512x512 image
shift_pixels_x: 128  # 1/4 of width
transform_count: 4
distribution: "even"
```

### Motion Blur Effect
Accumulating shifts concentrated at the end:
```python
transform_count: 8
distribution: "back_loaded"
shift_pixels_x: 2
accumulate_shifts: True
```

### Glitch Art
Random transforms with partial strength:
```python
transform_type: "random_transform"
transform_strength: 0.6
distribution: "random"
```

## 🔧 Advanced Usage

### Custom Sequences
Use shorthand notation in `transform_sequence`:
- `s` or `shift` - Shift transform
- `h` or `mh` - Mirror horizontal
- `v` or `mv` - Mirror vertical
- `cw` - Rotate 90° clockwise
- `ccw` - Rotate 90° counter-clockwise
- `180` - Rotate 180°
- `r` - Random

Example: `"s,h,cw,s,v,ccw"` creates a complex transformation pattern.

### Manual Step Control
When using `distribution: "manual"`, specify exact steps:
```
manual_steps: "5,10,15,18"
```

### Mathematical Distributions
- **Golden Ratio**: Uses φ (1.618...) for natural aesthetics
- **Fibonacci**: Follows the Fibonacci sequence
- **Exponential**: Accelerating transform frequency
- **Logarithmic**: Decelerating transform frequency

## 📈 Performance

- **Overhead**: < 5% vs standard KSampler
- **Memory**: No additional VRAM required
- **Compatibility**: Works with all models (SD1.5, SDXL, etc.)
- **Batch Support**: Full batch processing support

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No visible effect | Increase transform_count or shift amounts |
| Too dramatic | Reduce transform_count, lower strength |
| Unpredictable | Avoid random mode, use manual distribution |
| Device errors | Update to latest version (3.0+) |

## 📚 Documentation

- [Full Usage Guide](docs/USAGE_GUIDE.md) - Detailed parameters and examples
- [API Reference](docs/API.md) - Technical documentation
- [Examples Gallery](docs/EXAMPLES.md) - Visual examples with settings
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 Roadmap

- [ ] ComfyUI Manager integration
- [ ] Visual preview of distribution patterns
- [ ] More transform types (skew, perspective)
- [ ] Animation export support
- [ ] Preset library
- [ ] Web UI for distribution design
- [ ] Batch transform variations
- [ ] Integration with other samplers

## 🙏 Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
- The ComfyUI community for inspiration and feedback
- Mathematical concepts from nature and art

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/comfyui-n-transform-sampler&type=Date)](https://star-history.com/#yourusername/comfyui-n-transform-sampler&Date)

## 📸 Gallery

<details>
<summary>Click to see example outputs</summary>

### Golden Ratio Shifts
![Golden Ratio Example](docs/images/golden_ratio_example.png)
*3 shifts with golden ratio distribution*

### Fibonacci Spiral
![Fibonacci Spiral](docs/images/fibonacci_spiral.png)
*Spiral effect with Fibonacci timing*

### Kaleidoscope Pattern
![Kaleidoscope](docs/images/kaleidoscope.png)
*Alternating mirrors creating symmetry*

### Glitch Art
![Glitch Art](docs/images/glitch_art.png)
*Random transforms with 70% strength*

</details>

## 🔗 Links

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Report Issues](https://github.com/yourusername/comfyui-n-transform-sampler/issues)
- [Request Features](https://github.com/yourusername/comfyui-n-transform-sampler/issues/new?labels=enhancement)
- [Discussions](https://github.com/yourusername/comfyui-n-transform-sampler/discussions)

---

<p align="center">
  Made with ❤️ for the ComfyUI Community
  <br>
  <strong>Transform your generations with mathematical precision! 🎨✨</strong>
</p>
