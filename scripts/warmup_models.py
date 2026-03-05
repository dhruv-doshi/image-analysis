"""Pre-download all model weights to HF_HOME. Run once at Docker build time."""

import os

os.environ.setdefault("HF_HOME", "/app/.hf_cache")

print("Warming up pyiqa models...")
import pyiqa

for name in ("brisque", "nima", "clipiqa+", "musiq"):
    try:
        pyiqa.create_metric(name, device="cpu")
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

print("Warming up rembg U²-Net...")
try:
    from rembg import remove
    from PIL import Image
    import numpy as np

    remove(Image.fromarray(np.zeros((4, 4, 3), dtype="uint8")))
    print("  ✓ rembg")
except Exception as e:
    print(f"  ✗ rembg: {e}")

print("Done.")
