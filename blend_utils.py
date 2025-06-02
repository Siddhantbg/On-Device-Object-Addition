import cv2
import numpy as np

def resize_background(bg_img, target_width=1024):
    """
    Resize background while maintaining aspect ratio.
    
    Args:
        bg_img: NumPy array of background image
        target_width: Desired width in pixels
        
    Returns:
        Resized background image
    """
    aspect_ratio = bg_img.shape[1] / bg_img.shape[0]
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(bg_img, (target_width, target_height))

def calculate_chair_size(bg_height):
    """
    Calculate realistic chair size based on background height.
    
    Args:
        bg_height: Height of the background image
        
    Returns:
        (width, height) tuple for chair dimensions
    """
    # Typical chair height is about 1/3 of room height
    chair_height = int(bg_height * 0.33)  # 33% of background height
    # Typical chair aspect ratio (width/height) is about 0.8
    chair_width = int(chair_height * 0.8)
    return chair_width, chair_height

def blend_object_with_mask(bg_img, obj_img, x_offset, y_offset, new_w, new_h):
    """
    Blend an object into a background image with proper mask handling and shape checking.
    
    Args:
        bg_img: a NumPy array (BGR) of the background (any size)
        obj_img: a NumPy array (BGR or BGRA) of the object to insert
        x_offset, y_offset: top-left corner where to place the object in bg_img
        new_w, new_h: desired width and height of the object in the background
    
    Returns:
        The composited bg_img (NumPy array) with the object blended in
    """
    # Extract alpha mask from obj_img
    if obj_img.shape[2] == 4:
        # RGBA image - split into RGB and alpha
        rgb_obj = obj_img[:, :, :3]
        alpha = obj_img[:, :, 3]
    else:
        # RGB image - create mask via thresholding
        rgb_obj = obj_img  # (still BGR)
        gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Resize rgb_obj and alpha to (new_w, new_h)
    obj_resized = cv2.resize(rgb_obj, (new_w, new_h))
    mask_resized = cv2.resize(alpha, (new_w, new_h))

    # Extract ROI from bg_img
    # First check if the ROI would exceed image boundaries
    if (x_offset < 0 or y_offset < 0 or 
        x_offset + new_w > bg_img.shape[1] or 
        y_offset + new_h > bg_img.shape[0]):
        raise ValueError(
            f"ROI coordinates ({x_offset}, {y_offset}) with size {new_w}x{new_h} "
            f"exceed background dimensions {bg_img.shape[1]}x{bg_img.shape[0]}"
        )
    
    roi = bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

    # Print shapes for debugging
    print("roi.shape:", roi.shape)                    # should be (new_h, new_w, 3)
    print("obj_resized.shape:", obj_resized.shape)    # (new_h, new_w, 3)
    print("mask_resized.shape:", mask_resized.shape)  # (new_h, new_w)

    # Ensure mask_resized is single-channel and dtype=uint8
    if len(mask_resized.shape) == 3:
        mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
    mask_resized = mask_resized.astype(np.uint8)

    # Verify shapes match before blending
    if roi.shape[:2] != mask_resized.shape:
        raise ValueError(f"ROI size {roi.shape[:2]} does not match mask size {mask_resized.shape}")
    if obj_resized.shape[:2] != mask_resized.shape:
        raise ValueError(f"Object size {obj_resized.shape[:2]} does not match mask size {mask_resized.shape}")

    # Blend using masks
    inv_mask = cv2.bitwise_not(mask_resized)                                # (new_h, new_w)
    bg_part = cv2.bitwise_and(roi, roi, mask=inv_mask)                     # (new_h, new_w, 3)
    fg_part = cv2.bitwise_and(obj_resized, obj_resized, mask=mask_resized) # (new_h, new_w, 3)
    composite_patch = cv2.add(bg_part, fg_part)                            # (new_h, new_w, 3)

    # Create a copy of bg_img to avoid modifying the input
    result = bg_img.copy()
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = composite_patch
    
    return result

def test_blend():
    """Test function to demonstrate usage"""
    # Load images
    bg = cv2.imread("background_scene.jpg")
    obj = cv2.imread("object.png", cv2.IMREAD_UNCHANGED)  # -1 flag to load alpha
    
    # Define placement parameters
    x, y = 100, 100
    w, h = 200, 200
    
    try:
        result = blend_object_with_mask(bg, obj, x, y, w, h)
        cv2.imwrite("result.png", result)
        print("Blending successful! Check result.png")
    except Exception as e:
        print(f"Error during blending: {str(e)}")

if __name__ == "__main__":
    test_blend() 