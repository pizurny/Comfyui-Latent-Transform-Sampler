"""
Latent Transform Sampler for ComfyUI
Applies N transformations (shift, mirror, rotate) at strategically distributed steps during sampling.
Full implementation with all distribution strategies and transform types.
"""

import torch
import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview

print("[Latent Transform] Loading version 3.1...")

class NTransformSampler:
    """
    Advanced sampler that applies N transformations at distributed intervals.
    Supports shifting, mirroring, and rotation with multiple distribution strategies.
    """
    
    def __init__(self):
        self.transform_state = {
            'applied_transforms': [],
            'scheduled_steps': [],
            'current_step': 0,
            'cumulative_shift_x': 0,
            'cumulative_shift_y': 0,
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Standard sampling parameters
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Transform configuration
                "transform_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Number of transformations to apply (0 = disabled)"
                }),
                
                "transform_type": ([
                    "shift",
                    "mirror_horizontal",
                    "mirror_vertical",
                    "rotate_90_cw",
                    "rotate_90_ccw",
                    "rotate_180",
                    "random_transform",
                    "sequence",
                    "alternate_mirrors",
                    "spiral",
                ], {
                    "default": "shift",
                    "tooltip": "Type of transformation to apply"
                }),
                
                "distribution": ([
                    "even",
                    "front_loaded",
                    "back_loaded",
                    "edges",
                    "center",
                    "golden_ratio",
                    "fibonacci",
                    "exponential",
                    "logarithmic",
                    "random",
                    "manual",
                ], {
                    "default": "manual",
                    "tooltip": "How to distribute transforms across steps"
                }),
                
                # Shift parameters
                "shift_pixels_x": ("INT", {
                    "default": 128,
                    "min": -512,
                    "max": 512,
                    "tooltip": "Horizontal shift in pixels (for shift transform)"
                }),
                "shift_pixels_y": ("INT", {
                    "default": 0,
                    "min": -512,
                    "max": 512,
                    "tooltip": "Vertical shift in pixels (for shift transform)"
                }),
                
                # Advanced options
                "min_spacing": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Minimum steps between transforms"
                }),
                
                "transform_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Strength of transformation (for gradual effects)"
                }),
                
                "accumulate_shifts": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether shifts should accumulate or reset each time"
                }),
                
                "reverse_on_second_half": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse transformations in the second half"
                }),
            },
            "optional": {
                "transform_sequence": ("STRING", {
                    "default": "shift,mirror_h,rotate_90_cw",
                    "multiline": False,
                    "tooltip": "Comma-separated sequence (if type='sequence')"
                }),
                
                "manual_steps": ("STRING", {
                    "default": "3,6,9",
                    "multiline": False,
                    "tooltip": "Comma-separated step numbers (if distribution='manual')"
                }),
                
                "debug_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print debug information to console"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "sampling/transform"
    DESCRIPTION = "Applies N transformations at distributed intervals during sampling"
    
    def apply_transform(self, latent: torch.Tensor, transform_type: str, 
                       strength: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Apply specified transformation to latent tensor.
        
        Args:
            latent: Input tensor [B, C, H, W]
            transform_type: Type of transformation
            strength: Strength factor (0-1)
            **kwargs: Additional parameters (shift amounts, etc.)
        """
        device = latent.device
        dtype = latent.dtype
        
        # Ensure we're working with the right device
        latent = latent.to(device)
        
        if transform_type == "shift":
            shift_x = kwargs.get('shift_x', 0)
            shift_y = kwargs.get('shift_y', 0)
            
            # Apply strength factor
            shift_x = int(shift_x * strength)
            shift_y = int(shift_y * strength)
            
            # Convert pixels to latent units (8:1 ratio)
            latent_shift_x = shift_x // 8
            latent_shift_y = shift_y // 8
            
            if latent_shift_x != 0 or latent_shift_y != 0:
                return torch.roll(latent, shifts=(latent_shift_y, latent_shift_x), dims=(2, 3))
        
        elif transform_type == "mirror_horizontal" or transform_type == "mirror_h":
            if strength >= 0.5:  # Binary operation
                return torch.flip(latent, dims=[3])
            else:
                # Gradual blend with original
                flipped = torch.flip(latent, dims=[3])
                return latent * (1 - strength) + flipped * strength
        
        elif transform_type == "mirror_vertical" or transform_type == "mirror_v":
            if strength >= 0.5:
                return torch.flip(latent, dims=[2])
            else:
                flipped = torch.flip(latent, dims=[2])
                return latent * (1 - strength) + flipped * strength
        
        elif transform_type == "rotate_90_cw":
            if strength >= 0.5:
                return torch.rot90(latent, k=-1, dims=(2, 3))
            else:
                rotated = torch.rot90(latent, k=-1, dims=(2, 3))
                return latent * (1 - strength) + rotated * strength
        
        elif transform_type == "rotate_90_ccw":
            if strength >= 0.5:
                return torch.rot90(latent, k=1, dims=(2, 3))
            else:
                rotated = torch.rot90(latent, k=1, dims=(2, 3))
                return latent * (1 - strength) + rotated * strength
        
        elif transform_type == "rotate_180":
            if strength >= 0.5:
                return torch.rot90(latent, k=2, dims=(2, 3))
            else:
                rotated = torch.rot90(latent, k=2, dims=(2, 3))
                return latent * (1 - strength) + rotated * strength
        
        elif transform_type == "random_transform":
            # Randomly choose a transform
            transforms = ["shift", "mirror_horizontal", "mirror_vertical", 
                        "rotate_90_cw", "rotate_90_ccw", "rotate_180"]
            chosen = random.choice(transforms)
            return self.apply_transform(latent, chosen, strength, **kwargs)
        
        return latent
    
    def calculate_transform_steps(self, total_steps: int, n_transforms: int, 
                                 distribution: str, min_spacing: int = 2,
                                 manual_steps: str = "") -> List[int]:
        """
        Calculate which steps to apply transforms based on distribution strategy.
        
        Args:
            total_steps: Total number of sampling steps
            n_transforms: Number of transforms to apply
            distribution: Distribution strategy
            min_spacing: Minimum spacing between transforms
            manual_steps: Manual step specification (comma-separated)
        
        Returns:
            List of step indices where transforms should be applied
        """
        if n_transforms == 0:
            return []
        
        if distribution == "manual" and manual_steps:
            # Parse manual steps
            try:
                steps = [int(s.strip()) for s in manual_steps.split(",")]
                # Filter valid steps
                steps = [s for s in steps if 0 <= s < total_steps]
                return sorted(steps[:n_transforms])
            except:
                print(f"[Latent Transform] Error parsing manual steps, using even distribution")
                distribution = "even"
        
        # Single transform optimization
        if n_transforms == 1:
            if distribution in ["front_loaded", "edges"]:
                return [min(min_spacing, total_steps - 1)]
            elif distribution in ["back_loaded"]:
                return [max(total_steps - min_spacing, 0)]
            elif distribution == "center":
                return [total_steps // 2]
            else:
                return [total_steps // 2]
        
        # Multiple transforms
        if distribution == "even":
            # Evenly distributed
            if n_transforms >= total_steps:
                return list(range(total_steps))
            interval = total_steps / (n_transforms + 1)
            return [int(interval * (i + 1)) for i in range(n_transforms)]
        
        elif distribution == "front_loaded":
            # More transforms early
            steps = []
            position = min_spacing
            for i in range(n_transforms):
                if position >= total_steps:
                    break
                steps.append(position)
                # Exponentially increasing intervals
                position += int((2 ** (i * 0.5)) * min_spacing)
            return steps[:n_transforms]
        
        elif distribution == "back_loaded":
            # More transforms late
            front_steps = self.calculate_transform_steps(
                total_steps, n_transforms, "front_loaded", min_spacing
            )
            return sorted([total_steps - 1 - s for s in front_steps if total_steps - 1 - s >= 0])
        
        elif distribution == "edges":
            # Cluster at start and end
            if n_transforms == 2:
                return [min_spacing, total_steps - min_spacing - 1]
            
            half = n_transforms // 2
            start_steps = []
            end_steps = []
            
            for i in range(half):
                start_steps.append(min_spacing + i * min_spacing)
            
            for i in range(n_transforms - half):
                end_steps.append(total_steps - 1 - (i * min_spacing))
            
            return sorted(start_steps + end_steps)[:n_transforms]
        
        elif distribution == "center":
            # Cluster around center
            center = total_steps // 2
            steps = []
            for i in range(n_transforms):
                offset = (i - n_transforms // 2) * min_spacing
                step = center + offset
                if 0 <= step < total_steps:
                    steps.append(step)
            return sorted(steps)
        
        elif distribution == "golden_ratio":
            # Golden ratio distribution
            phi = 1.618033988749895
            steps = []
            for i in range(n_transforms):
                # Use golden ratio for positioning
                position = int(total_steps * (i / phi) % total_steps)
                steps.append(position)
            return sorted(list(set(steps)))[:n_transforms]
        
        elif distribution == "fibonacci":
            # Fibonacci sequence positions
            if n_transforms == 1:
                return [total_steps // 2]
            
            # Generate Fibonacci sequence
            fib = [1, 1]
            while len(fib) < n_transforms and fib[-1] < total_steps:
                fib.append(fib[-1] + fib[-2])
            
            # Scale to fit in total_steps
            if fib[-1] > total_steps:
                scale = (total_steps - min_spacing * 2) / fib[-1]
                steps = [int(min_spacing + f * scale) for f in fib[:n_transforms]]
            else:
                steps = [f for f in fib if f < total_steps][:n_transforms]
            
            return sorted(steps)
        
        elif distribution == "exponential":
            # Exponential distribution
            steps = []
            for i in range(n_transforms):
                # Exponential curve
                t = i / max(n_transforms - 1, 1)
                position = int((math.exp(t * 2) - 1) / (math.exp(2) - 1) * (total_steps - 1))
                steps.append(position)
            return sorted(list(set(steps)))
        
        elif distribution == "logarithmic":
            # Logarithmic distribution
            steps = []
            for i in range(n_transforms):
                # Logarithmic curve
                t = (i + 1) / (n_transforms + 1)
                position = int(math.log(1 + t * (math.e - 1)) * (total_steps - 1))
                steps.append(position)
            return sorted(list(set(steps)))
        
        elif distribution == "random":
            # Random with minimum spacing constraint
            if total_steps < n_transforms * min_spacing:
                # Not enough space, just distribute evenly
                return self.calculate_transform_steps(
                    total_steps, n_transforms, "even", 1
                )
            
            possible_steps = list(range(min_spacing, total_steps - min_spacing))
            steps = []
            
            for _ in range(n_transforms):
                if not possible_steps:
                    break
                
                step = random.choice(possible_steps)
                steps.append(step)
                
                # Remove nearby steps to maintain spacing
                possible_steps = [
                    s for s in possible_steps 
                    if abs(s - step) >= min_spacing
                ]
            
            return sorted(steps)
        
        # Default fallback
        return self.calculate_transform_steps(total_steps, n_transforms, "even", min_spacing)
    
    def parse_sequence(self, sequence_str: str) -> List[str]:
        """Parse transform sequence from string."""
        if not sequence_str:
            return []
        
        transforms = []
        for t in sequence_str.split(','):
            t = t.strip().lower()
            # Map shortcuts
            if t in ['h', 'mirror_h', 'mh']:
                transforms.append('mirror_horizontal')
            elif t in ['v', 'mirror_v', 'mv']:
                transforms.append('mirror_vertical')
            elif t in ['cw', 'r90cw', '90cw']:
                transforms.append('rotate_90_cw')
            elif t in ['ccw', 'r90ccw', '90ccw']:
                transforms.append('rotate_90_ccw')
            elif t in ['180', 'r180']:
                transforms.append('rotate_180')
            elif t in ['s', 'shift']:
                transforms.append('shift')
            elif t in ['r', 'random']:
                transforms.append('random_transform')
            else:
                transforms.append(t)
        
        return transforms
    
    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
              sampler_name, scheduler, denoise, transform_count, transform_type,
              distribution, shift_pixels_x, shift_pixels_y, min_spacing,
              transform_strength, accumulate_shifts, reverse_on_second_half,
              transform_sequence="", manual_steps="", debug_mode=True):
        """
        Main sampling function with N transforms at distributed intervals.
        """
        # Initialize state
        self.transform_state = {
            'applied_transforms': [],
            'scheduled_steps': [],
            'current_step': 0,
            'cumulative_shift_x': 0 if accumulate_shifts else 0,
            'cumulative_shift_y': 0 if accumulate_shifts else 0,
            'sequence_index': 0,
            'reversed': False,
        }
        
        # Calculate transform steps
        transform_steps = self.calculate_transform_steps(
            steps, transform_count, distribution, min_spacing, manual_steps
        )
        self.transform_state['scheduled_steps'] = transform_steps
        
        # Parse sequence if needed
        sequence = []
        if transform_type == "sequence":
            sequence = self.parse_sequence(transform_sequence)
            if not sequence:
                sequence = ["shift"]  # Default fallback
        elif transform_type == "alternate_mirrors":
            sequence = ["mirror_horizontal", "mirror_vertical"] * (transform_count // 2 + 1)
        elif transform_type == "spiral":
            sequence = ["rotate_90_cw", "shift"] * (transform_count // 2 + 1)
        
        if debug_mode:
            print(f"\n[Latent Transform] Configuration:")
            print(f"  Transform Count: {transform_count}")
            print(f"  Transform Type: {transform_type}")
            print(f"  Distribution: {distribution}")
            print(f"  Scheduled Steps: {transform_steps}")
            if sequence:
                print(f"  Sequence: {sequence[:transform_count]}")
            print(f"  Total Steps: {steps}")
            print(f"  Accumulate Shifts: {accumulate_shifts}")
            print(f"  Reverse Second Half: {reverse_on_second_half}")
        
        # Create callback function
        def transform_callback(step, x0, x, total_steps):
            """Callback executed at each sampling step."""
            
            self.transform_state['current_step'] = step
            
            # Check if this step should have a transform
            if step in transform_steps:
                transform_index = transform_steps.index(step)
                
                # Check if we should reverse (second half)
                if reverse_on_second_half and step >= total_steps // 2:
                    if not self.transform_state['reversed']:
                        self.transform_state['reversed'] = True
                        if accumulate_shifts:
                            # Reverse accumulated shifts
                            self.transform_state['cumulative_shift_x'] *= -1
                            self.transform_state['cumulative_shift_y'] *= -1
                
                # Determine which transform to apply
                if transform_type == "sequence" or transform_type == "alternate_mirrors" or transform_type == "spiral":
                    current_transform = sequence[transform_index % len(sequence)]
                else:
                    current_transform = transform_type
                
                # Calculate dynamic strength if needed
                strength = transform_strength
                if distribution in ["front_loaded", "back_loaded"]:
                    # Adjust strength based on position
                    progress = step / total_steps
                    if distribution == "front_loaded":
                        strength *= (1 - progress * 0.5)  # Stronger early
                    else:
                        strength *= (0.5 + progress * 0.5)  # Stronger late
                
                # Prepare kwargs for transform
                kwargs = {}
                if current_transform == "shift":
                    if accumulate_shifts:
                        self.transform_state['cumulative_shift_x'] += shift_pixels_x
                        self.transform_state['cumulative_shift_y'] += shift_pixels_y
                        kwargs['shift_x'] = self.transform_state['cumulative_shift_x']
                        kwargs['shift_y'] = self.transform_state['cumulative_shift_y']
                    else:
                        kwargs['shift_x'] = shift_pixels_x
                        kwargs['shift_y'] = shift_pixels_y
                
                # Apply transform
                try:
                    x_transformed = self.apply_transform(x, current_transform, strength, **kwargs)
                    x.copy_(x_transformed)
                    
                    # Log the transform
                    self.transform_state['applied_transforms'].append({
                        'step': step,
                        'transform': current_transform,
                        'strength': strength,
                        'kwargs': kwargs
                    })
                    
                    if debug_mode:
                        if current_transform == "shift" and accumulate_shifts:
                            print(f"[Latent Transform] Step {step}: {current_transform} "
                                  f"(cumulative: {kwargs['shift_x']}, {kwargs['shift_y']} pixels)")
                        else:
                            print(f"[Latent Transform] Step {step}: {current_transform} "
                                  f"(strength: {strength:.2f})")
                
                except Exception as e:
                    print(f"[Latent Transform] Error applying {current_transform} at step {step}: {e}")
            
            return x
        
        # Get device
        device = comfy.model_management.get_torch_device()
        
        # Prepare latent
        latent = latent_image["samples"].to(device)
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        
        # Prepare noise
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent, seed, batch_inds)
        
        # Get noise mask if present
        noise_mask = latent_image.get("noise_mask", None)
        if noise_mask is not None:
            noise_mask = noise_mask.to(device)
        
        # Combine with preview callback
        preview_callback = latent_preview.prepare_callback(model, steps)
        
        def combined_callback(step, x0, x, total_steps):
            x = transform_callback(step, x0, x, total_steps)
            if preview_callback:
                preview_callback(step, x0, x, total_steps)
            return x
        
        # Sample with our callback
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=True,
            noise_mask=noise_mask,
            callback=combined_callback,
            disable_pbar=disable_pbar,
            seed=seed
        )
        
        # Prepare output
        out = latent_image.copy()
        out["samples"] = samples
        
        if debug_mode:
            print(f"\n[Latent Transform] Sampling complete!")
            print(f"  Applied {len(self.transform_state['applied_transforms'])} transforms")
            if accumulate_shifts and any(t['transform'] == 'shift' for t in self.transform_state['applied_transforms']):
                print(f"  Final shift: ({self.transform_state['cumulative_shift_x']}, "
                      f"{self.transform_state['cumulative_shift_y']}) pixels")
        
        return (out,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NTransformSampler": NTransformSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NTransformSampler": "Latent Transform Sampler 🔄",
}

print("[Latent Transform] Node registered successfully!")