import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from advanced_blend import advanced_blend
import speech_recognition as sr

def listen_and_recognize() -> str:
    """Capture and recognize speech from microphone."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.write("🎙️ Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        st.write("🎙️ Listening... (speak your placement command)")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio. Please try again.")
        return ""
    except sr.RequestError:
        st.error("❌ Could not reach speech recognition service. Check your internet connection.")
        return ""

def parse_command_to_coords(command: str, bg_w: int, bg_h: int, obj_w: int, obj_h: int) -> tuple[int, int]:
    """Parse voice command into x,y coordinates."""
    # Default to center if command is empty or unrecognized
    x = (bg_w - obj_w) // 2
    y = (bg_h - obj_h) // 2
    
    if "top" in command:
        y = 0
        if "left" in command:
            x = 0
        elif "right" in command:
            x = bg_w - obj_w
    elif "bottom" in command:
        y = bg_h - obj_h
        if "left" in command:
            x = 0
        elif "right" in command:
            x = bg_w - obj_w
    elif "center" in command or "middle" in command:
        x = (bg_w - obj_w) // 2
        y = (bg_h - obj_h) // 2
    
    return x, y

# Set page config
st.set_page_config(
    page_title="Object Placement Tool",
    page_icon="🪑",
    layout="wide"
)

st.title("Touch & Voice Object Placement")
st.markdown("""
This tool lets you place objects into scenes with realistic shadows and blending.
1. Upload a background image
2. Upload an object with transparency (PNG)
3. Adjust the object size
4. Either:
   - Click on the background to place the object, or
   - Use voice commands like "place in top left" or "move to bottom right"
""")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    # Background image upload
    bg_file = st.file_uploader("Upload background image", type=["png", "jpg", "jpeg"])

with col2:
    # Object image upload
    obj_file = st.file_uploader("Upload object PNG with transparency", type=["png"])

if bg_file and obj_file:
    # Load full-resolution background as PIL & get its size
    bg_pil_full = Image.open(bg_file).convert("RGB")
    bg_w_full, bg_h_full = bg_pil_full.size

    # Let user choose object scale as a percentage of background width
    scale_pct = st.slider("Object scale (%) relative to background width", 5, 100, 25, step=1)
    # Compute full-res object width = scale_pct% of bg_w_full
    new_w = int((scale_pct / 100.0) * bg_w_full)

    # Determine object aspect ratio from original object image
    obj_pil = Image.open(obj_file)
    if obj_pil.mode != "RGBA":
        obj_pil = obj_pil.convert("RGBA")
    obj_w_full, obj_h_full = obj_pil.size
    aspect = obj_h_full / obj_w_full
    # Compute new_h to maintain aspect ratio
    new_h = int(new_w * aspect)

    st.write(f"→ At {scale_pct}%, object will be {new_w}px wide × {new_h}px tall (full resolution).")

    # Determine preview dimensions (max width = 400px)
    preview_max_w = 400
    if bg_w_full > preview_max_w:
        scale = preview_max_w / float(bg_w_full)
        preview_w = preview_max_w
        preview_h = int(bg_h_full * scale)
    else:
        scale = 1.0
        preview_w = bg_w_full
        preview_h = bg_h_full

    # Create a resized copy for the user to click on
    bg_pil_display = bg_pil_full.resize((preview_w, preview_h), Image.LANCZOS)
    
    # Add voice command option
    st.header("📣 Voice Placement")
    st.write("Try commands like: 'place in top left', 'move to bottom right', 'put in center'")
    
    voice_col1, voice_col2 = st.columns([1, 2])
    with voice_col1:
        if st.button("🎙️ Record Placement Command"):
            command = listen_and_recognize()
            if command:
                st.write(f"Recognized: '{command}'")
                x_full, y_full = parse_command_to_coords(command, bg_w_full, bg_h_full, new_w, new_h)
                x_disp = int(x_full * scale)
                y_disp = int(y_full * scale)
                st.session_state['voice_coords'] = (x_disp, y_disp)
                st.experimental_rerun()
    
    st.write("👆 Or click on the image below to choose placement. A semi-transparent red circle will show your selection.")
    
    # Create columns to constrain image width
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    
    with col_mid:
        # Get coordinates either from voice or click
        coords = streamlit_image_coordinates(bg_pil_display, key="bg_clicker")
        x_disp, y_disp = None, None
        
        # Check for voice coordinates first
        if 'voice_coords' in st.session_state:
            x_disp, y_disp = st.session_state['voice_coords']
            del st.session_state['voice_coords']  # Clear after use
        # Fall back to click coordinates
        elif coords:
            x_disp = coords["x"]
            y_disp = coords["y"]

        # Draw a translucent red circle at the location
        preview = bg_pil_display.convert("RGBA")
        overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if x_disp is not None and y_disp is not None:
            # Map displayed coords to full-res coords
            x_full = int(x_disp / scale)
            y_full = int(y_disp / scale)
            # Circle radius = 20px in full-res → 20 * scale in preview
            rad_disp = int(20 * scale)
            draw.ellipse(
                [(x_disp - rad_disp, y_disp - rad_disp),
                 (x_disp + rad_disp, y_disp + rad_disp)],
                fill=(255, 0, 0, 100)  # translucent red
            )
            
            # Add crosshair lines
            line_length = rad_disp
            # Vertical line
            draw.line([(x_disp, y_disp - line_length), (x_disp, y_disp + line_length)], 
                     fill=(255, 255, 255, 180), width=2)
            # Horizontal line
            draw.line([(x_disp - line_length, y_disp), (x_disp + line_length, y_disp)], 
                     fill=(255, 255, 255, 180), width=2)

        # Composite overlay onto preview
        preview = Image.alpha_composite(preview, overlay).convert("RGB")

        # Show the preview at fixed preview width
        st.image(preview, width=preview_w)

        # Compute x_offset, y_offset in full resolution
        if x_disp is not None and y_disp is not None:
            # Bottom-center alignment: object bottom center at (x_full, y_full)
            x_offset = x_full - (new_w // 2)
            y_offset = y_full - new_h
        else:
            # Default: center of background
            x_offset = (bg_w_full - new_w) // 2
            y_offset = (bg_h_full - new_h) // 2
    
    # Create columns for the button and debug info
    col5, col6 = st.columns([1, 2])
    
    with col5:
        # Place object button with loading indicator
        place_button = st.button("Place Object Here", type="primary")
    
    if place_button and x_disp is not None and y_disp is not None:
        with st.spinner("Compositing object..."):
            # Convert PIL to cv2 format for advanced_blend
            bg_img = cv2.cvtColor(np.array(bg_pil_full), cv2.COLOR_RGB2BGR)
            
            # Load object with alpha
            obj_rgba = np.array(obj_pil)
            
            # Run the blend on full resolution images
            final_img = advanced_blend(
                bg_img=bg_img,
                obj_img=obj_rgba,
                x_offset=x_offset,
                y_offset=y_offset,
                new_w=new_w,
                new_h=new_h,
                debug=True
            )
            
            # Convert back to PIL for display
            final_pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
            
            # Show result at column width (auto-scaled)
            st.image(final_pil, caption="Final Composite", use_container_width=True)
            
            # Show debug info in an expander
            with st.expander("Show placement details"):
                st.write(f"Preview coordinates: ({x_disp}, {y_disp})")
                st.write(f"Full-res coordinates: ({x_full}, {y_full})")
                st.write(f"Object offset: ({x_offset}, {y_offset})")
                st.write(f"Object size: {new_w}x{new_h} pixels")
                st.write(f"Object scale: {scale_pct}% of background width")
                st.write(f"Preview scale: {scale:.2f}")
                st.write(f"Background size: {bg_w_full}x{bg_h_full} pixels")
else:
    st.info("Please upload both a background image and an object to begin.") 