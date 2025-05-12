import os
import sys
import logging
import platform
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def find_files_recursively(filename: str, start_dir: Optional[Path] = None, max_levels: int = 3) -> List[str]:
    """
    Find files by recursively looking in the current directory and parent directories.
    
    Args:
        filename: Name of the file to search for
        start_dir: Starting directory (defaults to current working directory)
        max_levels: Maximum number of parent directories to check (default: 3)
    
    Returns:
        List of paths where the file was found
    """
    found_files = []
    
    if start_dir is None:
        start_dir = Path.cwd()
    
    def search_up(current_dir: Path, levels_left: int):
        # Look for the file in the current directory
        target_path = current_dir / filename
        if os.path.exists(target_path):
            found_files.append(str(target_path))
        
        # Stop if we've reached the maximum number of levels
        if levels_left <= 0:
            return
        
        # If we have a parent directory, search there next
        parent_dir = current_dir.parent
        if parent_dir != current_dir:  # Check if we've reached the root
            search_up(parent_dir, levels_left - 1)
    
    # Start the search from the specified directory
    search_up(start_dir, max_levels)
    
    return found_files

def find_dssatpro_file() -> str:
    """Find DSSATPRO file location with enhanced search capabilities."""
    try:
        # Check if we're on Windows or macOS
        is_windows = platform.system() == 'Windows'
        
        # Define file name based on OS
        file_name = 'DSSATPRO.V48' if is_windows else 'DSSATPRO.L48'
        
        # Log platform and search info
        logger.info(f"Platform: {platform.system()}, searching for: {file_name}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        
        
        # Increase search depth
        found_files = find_files_recursively(file_name, max_levels=5)
        if found_files:
            logger.info(f"Found {file_name} at: {found_files[0]}")
            return found_files[0]  # Return the first found file
        
        # Try to find file in same directory as DETAIL.CDE
        try:
            detail_path = find_detail_cde()
            detail_dir = os.path.dirname(detail_path)
            pro_path = os.path.join(detail_dir, file_name)
            if os.path.exists(pro_path):
                logger.info(f"Found {file_name} in same directory as DETAIL.CDE: {pro_path}")
                return pro_path
        except Exception as e:
            logger.warning(f"Could not check DETAIL.CDE directory: {str(e)}")
        
        # Check common locations
        common_locations = []
        if is_windows:
            common_locations = [
                "C:\\DSSAT48",
                "C:\\Program Files\\DSSAT48",
                "C:\\Program Files (x86)\\DSSAT48",
                os.path.join(os.environ.get('USERPROFILE', ''), 'DSSAT48')
            ]
        else:  # macOS or Linux
            common_locations = [
                "/Applications/DSSAT48",
                "/usr/local/DSSAT48",
                os.path.join(os.environ.get('HOME', ''), 'DSSAT48')
            ]
        
        logger.info(f"Checking common DSSAT installation locations: {common_locations}")
        for location in common_locations:
            file_path = os.path.join(location, file_name)
            if os.path.exists(file_path):
                logger.info(f"Found {file_name} in common location: {file_path}")
                return file_path
                
        # If we've found DETAIL.CDE but not DSSATPRO file, check for alternative file naming
        alternative_file_name = 'DSSATPRO.L48' if is_windows else 'DSSATPRO.V48'
        logger.info(f"Trying alternative file name: {alternative_file_name}")
        
        # Check if alternative file exists in common locations
        for location in common_locations:
            alt_file_path = os.path.join(location, alternative_file_name)
            if os.path.exists(alt_file_path):
                logger.warning(f"Found alternative {alternative_file_name} in: {alt_file_path}")
                logger.warning(f"Using alternative file - may need to be renamed to {file_name}")
                return alt_file_path
        
        # Last resort - create a fallback path to handle cases where the file might be missing
        if not is_windows:
            # For macOS, try to find where DETAIL.CDE is located and assume DSSATPRO should be there
            try:
                detail_file = find_detail_cde()
                dssat_dir = os.path.dirname(detail_file)
                logger.warning(f"DSSATPRO.L48 not found, but DETAIL.CDE found in {dssat_dir}")
                logger.warning(f"Creating fallback directory structure with DSSAT base at: {dssat_dir}")
                return os.path.join(dssat_dir, file_name)
            except:
                pass
        
        # If we get here, we couldn't find the file
        raise FileNotFoundError(f"Could not find {file_name} file. Please ensure DSSAT is installed correctly.")
    
    except Exception as e:
        logger.error(f"Error finding {file_name}: {str(e)}")
        raise
def get_dssat_base() -> str:
    """Get DSSAT base directory using recursive file search"""
    try:
        v48_path = find_dssatpro_file()
        
        # Get the directory containing the DSSATPRO file
        dssat_base = os.path.dirname(v48_path)
        
        # Verify installation
        if verify_dssat_installation(dssat_base):
            return dssat_base
        else:
            logger.warning(f"Found DSSAT path but missing required files: {dssat_base}")
            raise ValueError("Valid DSSAT installation not found")
        
    except Exception as e:
        logger.error(f"Error getting DSSAT base directory: {str(e)}")
        raise

def verify_dssat_installation(base_path: str) -> bool:
    """Verify that all required DSSAT files exist"""
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        required_files = ['DSSATPRO.V48', 'DETAIL.CDE', 'DSCSM048.EXE']
    else:  # macOS
        required_files = ['DSSATPRO.L48', 'DETAIL.CDE', 'DSCSM048']
    
    return all(os.path.exists(os.path.join(base_path, file)) for file in required_files)

def find_detail_cde() -> str:
    """Find DETAIL.CDE file using recursive search with enhanced logging."""
    try:
        # Add more verbose logging to debug executable issues
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Platform: {platform.system()}")
        
        
        
        # Increase search depth
        found_files = find_files_recursively('DETAIL.CDE', max_levels=5)
        if found_files:
            logger.info(f"Found DETAIL.CDE at: {found_files[0]}")
            return found_files[0]
        
        # Try additional common locations for DSSAT installation
        common_locations = []
        if platform.system() == 'Windows':
            common_locations = [
                "C:\\DSSAT48",
                "C:\\Program Files\\DSSAT48",
                "C:\\Program Files (x86)\\DSSAT48",
                os.path.join(os.environ.get('USERPROFILE', ''), 'DSSAT48')
            ]
        else:  # macOS or Linux
            common_locations = [
                "/Applications/DSSAT48",
                "/usr/local/DSSAT48",
                os.path.join(os.environ.get('HOME', ''), 'DSSAT48')
            ]
        
        logger.info(f"Checking common DSSAT installation locations: {common_locations}")
        for location in common_locations:
            file_path = os.path.join(location, 'DETAIL.CDE')
            if os.path.exists(file_path):
                logger.info(f"Found DETAIL.CDE in common location: {file_path}")
                return file_path
        
        # If we get here, we couldn't find the file
        logger.error("Could not find DETAIL.CDE in any location.")
        raise FileNotFoundError("Could not find DETAIL.CDE file. Please ensure DSSAT is installed correctly.")
    
    except Exception as e:
        logger.error(f"Error finding DETAIL.CDE: {str(e)}")
        raise

def get_crop_details() -> List[dict]:
    """Get crop codes, names, and directories from DETAIL.CDE and DSSATPRO file."""
    try:
        is_windows = platform.system() == 'Windows'
        
        # Find files using recursive search method
        detail_cde_path = find_detail_cde()
        dssatpro_path = find_dssatpro_file()
        
        crop_details = []
        in_crop_section = False
        
        # Step 1: Get crop codes and names from DETAIL.CDE
        with open(detail_cde_path, 'r') as file:
            for line in file:
                if '*Crop and Weed Species' in line:
                    in_crop_section = True
                    continue
                    
                if '@CDE' in line:
                    continue
                    
                if line.startswith('*') and in_crop_section:
                    break
                    
                if in_crop_section and line.strip():
                    crop_code = line[:8].strip()
                    crop_name = line[8:72].strip()
                    if crop_code and crop_name:
                        crop_details.append({
                            'code': crop_code[:2],
                            'name': crop_name,
                            'directory': ''
                        })
        
        # Step 2: Get directories from DSSATPRO file
        with open(dssatpro_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    folder_code = parts[0]
                    if folder_code.endswith('D'):
                        code = folder_code[:-1]
                        directory = parts[1].strip()
                        
                        # OS-specific directory formatting
                        if is_windows:
                            directory = directory.replace(': ', ':')
                        else:  # macOS
                            # Remove any extra spaces or colons
                            directory = directory.replace(': ', '').replace(':', '')
                            # Convert Windows path separators if needed
                            directory = directory.replace('\\', '/')
                            # Ensure the path starts with the correct base directory
                            dssat_base = os.path.dirname(dssatpro_path)
                            if not directory.startswith(dssat_base):
                                directory = os.path.join(dssat_base, os.path.basename(directory))
                            # Clean up any double slashes and normalize path
                            directory = os.path.normpath(directory).replace('//', '/')
                        
                        # Update matching crop directory
                        for crop in crop_details:
                            if crop['code'] == code:
                                crop['directory'] = directory
                                break
        
        # Log the results for debugging
        if not is_windows:
            logger.info("Crop details loaded:")
            for crop in crop_details:
                logger.info(f"Name: {crop['name']}, Code: {crop['code']}, Directory: {crop['directory']}")
        
        return crop_details
        
    except Exception as e:
        error_msg = f"Error getting crop details: {str(e)}"
        if not is_windows:
            logger.error(error_msg, exc_info=True)
        else:
            logger.error(error_msg)
        return []
        
def prepare_folders() -> List[str]:
    """List available folders based on DETAIL.CDE crop codes and names."""
    try:
        # Find DETAIL.CDE using recursive search
        detail_cde_path = find_detail_cde()
        valid_folders = []
        in_crop_section = False
        
        with open(detail_cde_path, 'r') as file:
            for line in file:
                if '*Crop and Weed Species' in line:
                    in_crop_section = True
                    continue
                
                if '@CDE' in line:
                    continue
                
                if line.startswith('*') and in_crop_section:
                    break
                
                if in_crop_section and line.strip():
                    crop_code = line[:8].strip()
                    crop_name = line[8:72].strip()
                    if crop_code and crop_name:
                        valid_folders.append(crop_name)
        
        return valid_folders
        
    except Exception as e:
        logger.error(f"Error preparing folders: {str(e)}")
        return []

def initialize_dssat_paths():
    """Initialize DSSAT paths and set global configuration variables."""
    try:
        import config
        
        # Find DSSAT base using recursive search
        dssat_base = get_dssat_base()
        print(f"DSSAT Base Directory: {dssat_base}")
        
        # Ensure directory exists
        os.makedirs(dssat_base, exist_ok=True)
        
        # Define executable name based on OS
        is_windows = platform.system() == 'Windows'
        dssat_exe = os.path.join(dssat_base, "DSCSM048.EXE" if is_windows else "DSCSM048")
        
        # Set global config variables
        config.DSSAT_BASE = dssat_base
        config.DSSAT_EXE = dssat_exe
        
        return dssat_base, dssat_exe
        
    except Exception as e:
        logger.error(f"Error initializing DSSAT paths: {str(e)}")
        raise
