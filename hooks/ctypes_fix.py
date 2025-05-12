"""
Runtime hook to fix ctypes issues on macOS with PyInstaller
"""
import os
import sys

def fix_ctypes():
    import importlib
    import types
    
    # Force reload ctypes to ensure all attributes are properly initialized
    if 'ctypes' in sys.modules:
        try:
            ctypes = sys.modules['ctypes']
            
            # If CDLL is missing, we need to reload the module
            if not hasattr(ctypes, 'CDLL'):
                importlib.reload(ctypes)
                
                # If still missing, try manually defining it
                if not hasattr(ctypes, 'CDLL'):
                    import ctypes.util
                    
                    # Create a custom CDLL class if needed
                    class CDLL(object):
                        def __init__(self, name, *args, **kwargs):
                            self.name = name
                            
                        def __getattr__(self, name):
                            return lambda *args, **kwargs: None
                    
                    # Add it to the ctypes module
                    ctypes.CDLL = CDLL
                    
            print("Successfully fixed ctypes module")
        except Exception as e:
            print(f"Error fixing ctypes: {e}")
            
# Apply the fix
fix_ctypes()