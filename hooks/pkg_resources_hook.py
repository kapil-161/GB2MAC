
# PyInstaller runtime hook to fix pkg_resources scanning issues
import os
import sys

def patch_pkg_resources():
    try:
        import pkg_resources
        
        # Get original working_set
        original_working_set = pkg_resources.working_set
        
        # Monkey patch the processing of paths in WorkingSet
        original_add_entry = original_working_set.add_entry
        
        def safe_add_entry(self, entry):
            # Skip temporary directories that might cause problems
            if isinstance(entry, str) and any(x in entry for x in ['/tmp/', '/temp/', '_MEI']):
                return
            try:
                original_add_entry(entry)
            except Exception:
                # If any error occurs during path scanning, just ignore this path
                pass
                
        # Apply the patch
        original_working_set.add_entry = lambda entry: safe_add_entry(original_working_set, entry)
        
        print("Successfully patched pkg_resources")
    except Exception as e:
        print(f"Error patching pkg_resources: {e}")
        
# Apply the patch
patch_pkg_resources()
