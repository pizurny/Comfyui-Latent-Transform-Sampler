# Version: 3.0 | Date: 2025-10-06 | Project: N-Transform Sampler | AI: Claude Opus 4.1

"""
N-Transform Sampler for ComfyUI
Advanced sampling with multiple transformation types and distribution strategies.
"""

print("[N-Transform] Initializing custom node package v3.0...")

# Try to import the main N-Transform sampler
try:
    from .n_transform_sampler import NTransformSampler
    print("[N-Transform] Main sampler loaded successfully")
    MAIN_LOADED = True
except Exception as e:
    print(f"[N-Transform] Warning: Could not load main sampler: {e}")
    MAIN_LOADED = False

# Optional: Also try to load the fixed version if it exists
try:
    from .latent_shift_fixed import LatentShiftSamplerFixed
    print("[N-Transform] Fixed shift sampler loaded")
    FIXED_LOADED = True
except:
    FIXED_LOADED = False

# Optional: Load simple version
try:
    from .latent_shift_simple import LatentShiftSamplerSimple, LatentShiftTest
    print("[N-Transform] Simple samplers loaded")
    SIMPLE_LOADED = True
except:
    SIMPLE_LOADED = False

# Build node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if MAIN_LOADED:
    NODE_CLASS_MAPPINGS["NTransformSampler"] = NTransformSampler
    NODE_DISPLAY_NAME_MAPPINGS["NTransformSampler"] = "N-Transform Sampler 🔄"

if FIXED_LOADED:
    NODE_CLASS_MAPPINGS["LatentShiftSamplerFixed"] = LatentShiftSamplerFixed
    NODE_DISPLAY_NAME_MAPPINGS["LatentShiftSamplerFixed"] = "Latent Shift (Fixed) ✓"

if SIMPLE_LOADED:
    NODE_CLASS_MAPPINGS["LatentShiftSamplerSimple"] = LatentShiftSamplerSimple
    NODE_CLASS_MAPPINGS["LatentShiftTest"] = LatentShiftTest
    NODE_DISPLAY_NAME_MAPPINGS["LatentShiftSamplerSimple"] = "Latent Shift (Simple) 🔄"
    NODE_DISPLAY_NAME_MAPPINGS["LatentShiftTest"] = "Latent Shift Test 🧪"

# Version info
__version__ = "3.0.0"
__author__ = "ComfyUI Custom Node Developer"

# Print summary
nodes_loaded = list(NODE_CLASS_MAPPINGS.keys())
print(f"[N-Transform] Version {__version__}")
print(f"[N-Transform] Loaded nodes: {', '.join(nodes_loaded) if nodes_loaded else 'None'}")

if not nodes_loaded:
    print("[N-Transform] ERROR: No nodes could be loaded! Check for import errors above.")

# Export
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']