import cv2
import numpy as np
from typing import Tuple, Optional
import os
from datetime import datetime

def create_debug_directory() -> str:
    """
    Create a timestamped directory for debug outputs.
    
    Returns:
        Path to the created debug directory
    """
    # Create base debug directory if it doesn't exist
    base_dir = "debug_outputs"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(base_dir, f"blend_{timestamp}")
    os.makedirs(debug_dir)
    
    return debug_dir

def extract_binary_mask(obj_img: np.ndarray, debug_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract BGR channels and create a strict binary mask from RGBA input.
    
    Args:
        obj_img: Input image (RGBA or RGB)
        debug_dir: Directory to save debug images
        
    Returns:
        Tuple of (bgr_channels, binary_mask)
    """
    def save_debug(name: str, img: np.ndarray) -> None:
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, name), img)
    
    # Split RGBA into BGR + alpha
    if obj_img.shape[2] == 4:
        # Convert RGB to BGR for the color channels
        obj_bgr = cv2.cvtColor(obj_img[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha_raw = obj_img[:, :, 3]
    else:
        obj_bgr = obj_img
        gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        alpha_raw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    
    # Create strict binary mask
    _, binary_mask = cv2.threshold(alpha_raw, 1, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    
    save_debug("debug_binary_mask_initial.png", binary_mask)
    save_debug("debug_bgr_initial.png", obj_bgr)
    
    return obj_bgr, binary_mask

def adjust_brightness(obj_img: np.ndarray, debug_dir: Optional[str] = None) -> np.ndarray:
    """
    Adjust brightness in HSV space to preserve black colors.
    
    Args:
        obj_img: Object image to adjust (BGR)
        debug_dir: Directory to save debug images
        
    Returns:
        Brightness-adjusted object image (BGR) with blacks preserved
    """
    def save_debug(name: str, img: np.ndarray) -> None:
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, name), img)
    
    # Convert to HSV and adjust V channel
    hsv = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # 5% brighter
    obj_bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    save_debug("debug_obj_bright.png", obj_bright)
    return obj_bright

def apply_perspective_transform(
    obj_img: np.ndarray,
    mask: np.ndarray,
    w: int,
    h: int,
    debug_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perspective transform with strict opacity control.
    
    Args:
        obj_img: Object image (BGR)
        mask: Binary mask (single channel)
        w, h: Width and height
        debug_dir: Directory to save debug images, if any
        
    Returns:
        Tuple of (warped object, warped mask)
    """
    def save_debug(name: str, img: np.ndarray) -> None:
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, name), img)
    
    # Compute transform matrix for 12% top narrowing
    delta = int(0.12 * w)
    src_pts = np.float32([[0,0], [w,0], [0,h], [w,h]])
    dst_pts = np.float32([[delta,0], [w-delta,0], [0,h], [w,h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply warping with appropriate interpolation
    obj_warp = cv2.warpPerspective(obj_img, M, (w, h), flags=cv2.INTER_AREA)
    mask_warp = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    
    # Force strict binary mask and dilate to cover antialiased edges
    _, mask_warp = cv2.threshold(mask_warp, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_warp = cv2.dilate(mask_warp, kernel, iterations=1)
    mask_warp = mask_warp.astype(np.uint8)
    save_debug("debug_mask_warp_final.png", mask_warp)
    
    # Create sanitized object with pure black background
    obj_sanitized = np.zeros_like(obj_warp)
    obj_sanitized[mask_warp == 255] = obj_warp[mask_warp == 255]
    save_debug("debug_obj_sanitized.png", obj_sanitized)
    
    # Verify mask is strictly binary
    unique_vals = np.unique(mask_warp)
    if not np.array_equal(unique_vals, np.array([0, 255])):
        raise ValueError(f"Warped mask is not strictly binary after dilation: {unique_vals}")
    
    return obj_sanitized, mask_warp

def create_cast_shadow(
    roi: np.ndarray,
    mask_warp: np.ndarray,
    h: int,
    w: int,
    debug_dir: Optional[str] = None
) -> np.ndarray:
    """
    Create and apply cast shadow directly to ROI.
    
    Args:
        roi: Background ROI to modify
        mask_warp: Binary mask
        h, w: Height and width
        debug_dir: Directory to save debug images, if any
        
    Returns:
        Modified ROI with shadow
    """
    def save_debug(name: str, img: np.ndarray) -> None:
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, name), img)
    
    # Create cast shadow (sheared)
    silhouette = (mask_warp > 0).astype(np.uint8) * 255
    M_shear = np.float32([[1, 0.15, 0], [0, 1, 5]])  # 15% right shear, 5px down
    sil_sheared = cv2.warpAffine(silhouette, M_shear, (w, h + 10))
    cast_shadow = cv2.GaussianBlur(sil_sheared, (151, 151), sigmaX=80)
    cast_shadow = (cast_shadow * 0.5).astype(np.uint8)  # 50% opacity
    
    # Apply cast shadow
    shadow_base = cv2.cvtColor(cast_shadow, cv2.COLOR_GRAY2BGR)
    y2, x2 = 5, 10
    h2, w2 = cast_shadow.shape[:2]
    
    if y2 + h2 <= h and x2 + w2 <= w:
        subroi = roi[y2:y2+h2, x2:x2+w2]
        temp = np.zeros_like(subroi)
        cv2.copyTo(shadow_base, cast_shadow, temp)
        roi[y2:y2+h2, x2:x2+w2] = cv2.add(subroi, temp)
    
    save_debug("debug_roi_with_shadow.png", roi)
    return roi

def advanced_blend(
    bg_img: np.ndarray,
    obj_img: np.ndarray,
    x_offset: int,
    y_offset: int,
    new_w: int,
    new_h: int,
    debug: bool = False
) -> np.ndarray:
    """
    Enhanced blending with guaranteed opacity and proper shadows.
    
    Args:
        bg_img: Background image (BGR)
        obj_img: Object image (BGR or BGRA)
        x_offset, y_offset: Position to place object
        new_w, new_h: Desired width and height
        debug: Whether to print debug information
        
    Returns:
        Composited image
    """
    # Create debug directory if needed
    debug_dir = create_debug_directory() if debug else None
    if debug:
        print(f"\nStarting advanced_blend with debug output in: {debug_dir}")
        print(f"Parameters: x={x_offset}, y={y_offset}, w={new_w}, h={new_h}")
        print(f"Shapes: bg={bg_img.shape}, obj={obj_img.shape}")
    
    # 1. Extract BGR and create strict binary mask
    obj_bgr, binary_mask = extract_binary_mask(obj_img, debug_dir)
    
    # 2. Resize with appropriate interpolation
    obj_resized = cv2.resize(obj_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(binary_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 3. Quick brightness adjustment in HSV space (preserves black)
    obj_bright = adjust_brightness(obj_resized, debug_dir)
    
    # 4. Extract and validate ROI
    h, w = new_h, new_w
    bg_h, bg_w = bg_img.shape[:2]
    if x_offset < 0 or y_offset < 0 or x_offset + w > bg_w or y_offset + h > bg_h:
        raise ValueError(f"Invalid ROI coordinates: ({x_offset}, {y_offset}) with size {w}x{h}")
    
    roi = bg_img[y_offset:y_offset+h, x_offset:x_offset+w].copy()
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "debug_roi_original.png"), roi)
    
    # 5. Apply perspective transform with sanitization
    obj_sanitized, mask_warp = apply_perspective_transform(obj_bright, mask_resized, w, h, debug_dir)
    
    # 6. Apply cast shadow
    roi_with_shadow = create_cast_shadow(roi, mask_warp, h, w, debug_dir)
    
    # 7. Final composite using cv2.copyTo
    # Verify mask is still strictly binary
    unique_vals = np.unique(mask_warp)
    if debug:
        print("Final mask_warp unique values:", unique_vals)
    if not np.array_equal(unique_vals, np.array([0, 255])):
        raise ValueError(f"mask_warp is not strictly binary before final composite: {unique_vals}")
    
    # Create final ROI and use cv2.copyTo
    final_roi = roi_with_shadow.copy()
    cv2.copyTo(obj_sanitized, mask_warp, final_roi)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "debug_final_roi_chair.png"), final_roi)
    
    # Write back to background
    bg_img[y_offset:y_offset+h, x_offset:x_offset+w] = final_roi
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "debug_full_composite.png"), bg_img)
    
    return bg_img 