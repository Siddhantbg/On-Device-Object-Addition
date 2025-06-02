import os
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
from typing import Tuple, Optional
import warnings
from blend_utils import blend_object_with_mask
warnings.filterwarnings("ignore")

class PhotorealisticObjectInsertion:
    def __init__(self, device: str = "cpu"):
        """
        Initialize the object insertion pipeline with CPU-based models.
        We use a lightweight Stable Diffusion inpainting model for edge blending
        and a shadow synthesis approach based on object mask transformation.
        """
        self.device = torch.device(device)
        
        # Load the inpainting model (small variant optimized for CPU)
        # Using stabilityai/stable-diffusion-2-inpainting (867M parameters)
        print("Loading inpainting model...")
        self.inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None  # Disable safety checker to save memory
        )
        self.inpainting_model.to(self.device)
        
        # Enable memory efficient attention
        self.inpainting_model.enable_attention_slicing()
        
        print("Models loaded successfully!")

    def preprocess_object(
        self, 
        object_image: Image.Image,
        target_size: Tuple[int, int] = (512, 512),
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the object image and create/refine its alpha mask.
        """
        # Convert PIL Image to numpy array
        obj_np = np.array(object_image)
        
        # If image has alpha channel, use it
        if obj_np.shape[-1] == 4:
            alpha_mask = obj_np[..., 3]
            obj_rgb = obj_np[..., :3]
        else:
            # If no alpha, use simple background removal
            obj_rgb = obj_np
            gray = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2GRAY)
            _, alpha_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Resize both object and mask
        obj_rgb = cv2.resize(obj_rgb, target_size)
        alpha_mask = cv2.resize(alpha_mask, target_size)
        
        # Refine the alpha mask
        kernel = np.ones((3,3), np.uint8)
        alpha_mask = cv2.erode(alpha_mask, kernel, iterations=1)
        alpha_mask = cv2.GaussianBlur(alpha_mask, (5,5), 0)
        
        return obj_rgb, alpha_mask

    def generate_shadow(
        self,
        alpha_mask: np.ndarray,
        light_direction: float = 45.0,
        shadow_length: float = 0.3,
        shadow_blur: int = 21,
        shadow_opacity: float = 0.6
        ) -> np.ndarray:
        """
        Generate a realistic shadow from the object's alpha mask.
        Uses perspective transform to simulate 3D shadow casting.
        """
        h, w = alpha_mask.shape
        
        # Create perspective transform matrix
        angle_rad = np.radians(light_direction)
        skew_x = shadow_length * np.cos(angle_rad)
        skew_y = shadow_length * np.sin(angle_rad)
        
        src_points = np.float32([[0,0], [w,0], [0,h], [w,h]])
        dst_points = np.float32([
            [skew_x*h, skew_y*h],
            [w+skew_x*h, skew_y*h],
            [0, h],
            [w, h]
        ])
        
        # Apply perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        shadow = cv2.warpPerspective(alpha_mask, transform_matrix, (w, h))
        
        # Blur and adjust opacity
        shadow = cv2.GaussianBlur(shadow, (shadow_blur, shadow_blur), 0)
        shadow = (shadow * shadow_opacity).astype(np.uint8)
        
        return shadow

    def blend_edges(
        self,
        scene: np.ndarray,
        object_rgb: np.ndarray,
        alpha_mask: np.ndarray,
        position: Tuple[int, int],
        blend_radius: int = 30
        ) -> np.ndarray:
        """
        Use inpainting model to seamlessly blend object edges with the scene.
        """
        with torch.no_grad():
            # Create a working area around the object
            mask_dilated = cv2.dilate(
                alpha_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blend_radius, blend_radius))
            )
            mask_edge = mask_dilated - alpha_mask
            
            # Convert to PIL images for the inpainting model
            scene_pil = Image.fromarray(scene)
            mask_pil = Image.fromarray(mask_edge)
            
            # Run inpainting
            blended = self.inpainting_model(
                prompt="high quality, photorealistic, detailed texture",
                image=scene_pil,
                mask_image=mask_pil,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            
            return np.array(blended)

    def insert_object(
        self,
        scene_path: str,
        object_path: str,
        position: Optional[Tuple[int, int]] = None,
        shadow_params: dict = None,
        output_path: str = "output.png"
        ) -> None:
        """
        Main function to insert an object into a scene with realistic shadows.
        """
        # Load images
        scene = Image.open(scene_path).convert("RGB")
        object_img = Image.open(object_path).convert("RGBA")
        
        # Convert to numpy
        scene_np = np.array(scene)
        obj_np = np.array(object_img)
        
        # If position not specified, place in center
        if position is None:
            position = (scene_np.shape[1]//2, scene_np.shape[0]//2)
        
        x_offset, y_offset = position
        
        # Get object dimensions
        new_h, new_w = obj_np.shape[:2]
        
        # Generate shadow
        shadow_params = shadow_params or {}
        shadow = self.generate_shadow(obj_np[..., 3] if obj_np.shape[-1] == 4 else cv2.cvtColor(obj_np, cv2.COLOR_RGB2GRAY), **shadow_params)
        
        # Apply shadow to scene
        scene_with_shadow = scene_np.copy()
        scene_with_shadow = cv2.subtract(scene_with_shadow, shadow[..., None])
        
        # Blend object using the new utility function
        final_composite = blend_object_with_mask(
            scene_with_shadow,
            obj_np,
            x_offset,
            y_offset,
            new_w,
            new_h
        )
        
        # Save result
        Image.fromarray(final_composite).save(output_path)
        print(f"Saved result to {output_path}")

def main():
    # Initialize the pipeline
    pipeline = PhotorealisticObjectInsertion()
    
    # Example usage
    scene_path = "background_scene.jpg"
    object_path = "object.png"
    
    # Custom shadow parameters
    shadow_params = {
        "light_direction": 45.0,
        "shadow_length": 0.3,
        "shadow_blur": 21,
        "shadow_opacity": 0.6
    }
    
    # Run the insertion
    pipeline.insert_object(
        scene_path=scene_path,
        object_path=object_path,
        shadow_params=shadow_params,
        output_path="output.png"
    )

if __name__ == "__main__":
    main() 