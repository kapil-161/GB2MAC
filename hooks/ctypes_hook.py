import os
import sys

def patch_ctypes():
    try:
        import ctypes
        import ctypes.util
        
        # Ensure CDLL is available
        if not hasattr(ctypes, 'CDLL'):
            # Try to reload the module
            import importlib
            importlib.reload(ctypes)
            
        print("Successfully patched ctypes")
    except Exception as e:
        print(f"Error patching ctypes: {e}")
        
# Apply the patch
patch_ctypes()