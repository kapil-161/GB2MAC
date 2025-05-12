
# PyInstaller runtime hook to fix OpenGL platform detection on macOS
import os
import sys

def patch_opengl():
    try:
        # Set environment variables before importing OpenGL
        os.environ["PYOPENGL_PLATFORM"] = "darwin"
        
        # Disable acceleration which can cause issues
        os.environ["PYOPENGL_ACCELERATE_DISABLE"] = "1"
        
        # Import OpenGL with these settings already in place
        import OpenGL
        OpenGL.ERROR_CHECKING = False
        
        print("Successfully configured OpenGL for macOS")
    except Exception as e:
        print(f"Error configuring OpenGL: {e}")
        
# Apply the patch
patch_opengl()
