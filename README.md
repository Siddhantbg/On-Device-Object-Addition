# Photorealistic Object Insertion Pipeline

This Python pipeline allows you to insert objects into background scenes with photorealistic results, including automatic shadow generation and edge blending. The pipeline runs entirely on CPU and uses pre-trained models for optimal performance.

## Features

- Automatic alpha mask creation/refinement for objects
- Realistic shadow generation with configurable parameters
- Edge blending using Stable Diffusion inpainting
- Runs entirely on CPU
- Memory-efficient processing

## Installation

1. Create a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your background scene image (e.g., `background_scene.jpg`) and object image (e.g., `object.png`) in the project directory.

2. Run the script:
```bash
python object_insertion.py
```

Or use the pipeline in your own code:

```python
from object_insertion import PhotorealisticObjectInsertion

pipeline = PhotorealisticObjectInsertion()

# Configure shadow parameters (optional)
shadow_params = {
    "light_direction": 45.0,  # Light direction in degrees
    "shadow_length": 0.3,     # Shadow length as proportion of object height
    "shadow_blur": 21,        # Shadow blur radius
    "shadow_opacity": 0.6     # Shadow opacity (0-1)
}

# Insert object
pipeline.insert_object(
    scene_path="background_scene.jpg",
    object_path="object.png",
    position=(500, 300),      # Optional: (x,y) position
    shadow_params=shadow_params,
    output_path="output.png"
)
```

## Models Used

- **Inpainting Model**: stabilityai/stable-diffusion-2-inpainting (867M parameters)
  - Optimized for CPU usage with attention slicing
  - Uses float32 precision for CPU compatibility

## Tips for Best Results

1. **Input Images**:
   - Background scene should be well-lit with clear lighting direction
   - Object image should have good contrast with its background
   - Higher resolution images will yield better results

2. **Shadow Parameters**:
   - Adjust `light_direction` to match the scene's lighting
   - Tune `shadow_opacity` based on scene brightness
   - Increase `shadow_blur` for softer shadows

3. **Memory Usage**:
   - The pipeline is optimized for CPU but may still require 8GB+ RAM
   - Processing time varies with image size and hardware

## Limitations

- Works best with objects that have clear boundaries
- Shadow generation assumes flat surface
- Processing time on CPU may be slower compared to GPU
- Maximum recommended image size: 1024x1024 pixels

## License

MIT License 