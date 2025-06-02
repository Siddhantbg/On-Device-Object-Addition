from object_insertion import PhotorealisticObjectInsertion

def test_object_insertion():
    # Initialize the pipeline
    print("Initializing pipeline (this will download the model on first run)...")
    pipeline = PhotorealisticObjectInsertion()
    
    # Configure parameters
    scene_path = "background_scene.jpg"  # Replace with your scene image path
    object_path = "object.png"           # Replace with your object image path
    
    # Shadow parameters - adjust these based on your scene
    shadow_params = {
        "light_direction": 45.0,    # Angle of light source (0-360 degrees)
        "shadow_length": 0.3,       # Shadow length (as proportion of object height)
        "shadow_blur": 21,          # Shadow blur radius
        "shadow_opacity": 0.6       # Shadow darkness (0-1)
    }
    
    # Optional: specify object position (x, y)
    # If not specified, object will be placed in center
    position = None  # or use (x, y) coordinates like (500, 300)
    
    print("Starting object insertion...")
    pipeline.insert_object(
        scene_path=scene_path,
        object_path=object_path,
        position=position,
        shadow_params=shadow_params,
        output_path="result.png"
    )
    print("Process complete! Check result.png")

if __name__ == "__main__":
    test_object_insertion() 