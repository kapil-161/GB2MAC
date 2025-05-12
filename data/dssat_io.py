import sys
import os
import pandas as pd
from io import StringIO


from typing import List, Optional

import logging
import glob
# Add project root to Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

"""
DSSAT file I/O operations
"""
# OPTIMIZED: Import only necessary pandas components
from pandas import DataFrame, concat, to_datetime, to_numeric, isna
# OPTIMIZED: Import only necessary numpy components
from numpy import nan
import logging
import subprocess
from typing import List, Optional
import config
from data.data_processing import standardize_dtypes, unified_date_convert
from utils.dssat_paths import get_crop_details

logger = logging.getLogger(__name__)

def prepare_experiment(selected_folder: str) -> List[tuple]:
    """List available experiments based on selected folder."""
    try:
        # Get crop details for directory and code
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details 
             if crop['name'].upper() == selected_folder.upper()),
            None
        )
        
        if not crop_info:
            logger.error(f"Could not find crop information for folder {selected_folder}")
            return []
            
        # Use directory from crop_info
        folder_path = crop_info['directory'].strip()
        if not folder_path:
            logger.error(f"No directory found for crop {selected_folder}")
            return []
            
        # Find X files using crop code
        x_file_pattern = f"*.{crop_info['code']}X"
        x_files = glob.glob(os.path.join(folder_path, x_file_pattern))
        
        result = []
        for file_path in x_files:
            filename = os.path.basename(file_path)
            exp_detail = filename  # Default to filename
            
            # Try to read experiment title from file
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        if "*EXP.DETAILS:" in line:
                            # Extract the title from the line
                            detail_part = line.strip().split("*EXP.DETAILS:")[1].strip()
                            exp_detail = ' '.join(detail_part.split()[1:])
                            break
            except Exception as e:
                logger.warning(f"Could not read experiment details from {filename}: {e}")
            
            # Add tuple of (display_name, filename) to the result
            result.append((exp_detail, filename))
        
        return result
        
    except Exception as e:
        logger.error(f"Error preparing experiments: {str(e)}")
        return []

def prepare_treatment(selected_folder: str, selected_experiment: str) -> Optional[DataFrame]:
    """Prepare treatment data based on selected folder and experiment."""
    try:
        # Get crop details
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details 
             if crop['name'].upper() == selected_folder.upper()),
            None
        )
        
        if not crop_info:
            logger.error(f"Could not find crop information for folder {selected_folder}")
            return None
            
        # Get directory and construct file path
        folder_path = crop_info['directory'].strip()
        if not folder_path:
            logger.error(f"No directory found for crop {selected_folder}")
            return None
            
        file_path = os.path.join(folder_path, selected_experiment)
        return read_treatments(file_path)
        
    except Exception as e:
        logger.error(f"Error preparing treatment: {str(e)}")
        return None

def read_treatments(file_path: str) -> Optional[DataFrame]:
    """Read and process treatment file."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Treatment file does not exist: {file_path}")
            return None
            
        with open(file_path, "r") as file:
            lines = file.readlines()
            
        # Find treatment section
        treatment_begins = next(
            (i for i, line in enumerate(lines) if line.startswith("*TREATMENT")),
            None
        )
        
        if treatment_begins is None:
            return None
            
        treatment_ends = next(
            (i for i, line in enumerate(lines) 
             if line.startswith("*") and i > treatment_begins),
            len(lines)
        )
        
        # Process treatment data
        treatment_data = lines[treatment_begins:treatment_ends]
        
        # Modified approach: keep lines that don't start with * or @ and aren't empty
        not_trash_lines = [
            line for line in treatment_data 
            if line.strip() and not line.strip().startswith(("*", "@","!")) 
            and len(line.strip().split()) > 1  # Ensure line has content
        ]
        
        # Create a dataframe with treatment info
        tr_list = []
        tname_list = []
        
        for line in not_trash_lines:
            parts = line.strip().split()
            if parts:
                tr = parts[0]  # First column is treatment number
                # Extract treatment name (column positions 9-36 in fixed width format)
                if len(line) > 9:
                    tname = line[9:36].strip()
                else:
                    # Fallback if line is too short
                    tname = " ".join(parts[1:]) if len(parts) > 1 else ""
                
                tr_list.append(tr)
                tname_list.append(tname)
        
        return DataFrame({
            "TR": tr_list,
            "TNAME": tname_list,
        })
        
    except Exception as e:
        logger.error(f"Error reading treatments: {str(e)}")
        return None

def prepare_out_files(selected_folder: str) -> List[str]:
    """List OUT files in the selected folder."""
    try:
        # Get crop details for directory
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details 
             if crop['name'].upper() == selected_folder.upper()),
            None
        )
        
        if not crop_info:
            logger.error(f"Could not find crop information for folder {selected_folder}")
            return []
            
        # Use directory from crop_info
        folder_path = crop_info['directory'].strip()
        if not folder_path:
            logger.error(f"No directory found for crop {selected_folder}")
            return []
            
        logger.info(f"Looking for OUT files in: {folder_path}")
        out_files = [f for f in os.listdir(folder_path) if f.endswith(".OUT")]
        logger.info(f"Output files found: {out_files}")
        return out_files
        
    except Exception as e:
        logger.error(f"Error preparing OUT files: {str(e)}")
        return []



def read_file(file_path: str) -> Optional[DataFrame]:
    """Read and process DSSAT output file with optimized performance.
    Handles both standard output files and FORAGE.OUT with special processing.
    """
    try:
        if os.path.basename(file_path) == file_path:  # File has no directory part
            # Try to find the file in crop directories
            crop_details = get_crop_details()
            for crop_info in crop_details:
                folder_path = crop_info['directory'].strip()
                possible_path = os.path.join(folder_path, file_path)
                if os.path.exists(possible_path):
                    file_path = possible_path
                    break
        # Normalize the file path to ensure the correct format
        file_path = os.path.normpath(file_path)
        print(f"Attempting to open file: {file_path}")
        print(f"File exists check: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None

        # Read file with efficient encoding handling
        encodings = ['utf-8', 'latin-1']
        lines = None
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    lines = file.readlines()
                break
            except UnicodeDecodeError:
                continue

        if not lines:
            logger.error(f"Could not read file with any encoding: {file_path}")
            return None

        # Check if this is FORAGE.OUT file
        is_forage_file = os.path.basename(file_path).upper() == "FORAGE.OUT"
        
        # Different processing paths for different file types
        if is_forage_file:
            # Special processing for FORAGE.OUT
            return process_forage_file(lines)
        else:
            # Standard processing for other DSSAT output files
            return process_standard_file(lines)

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def process_forage_file(lines: List[str]) -> Optional[DataFrame]:
    """Process FORAGE.OUT file using pandas with special handling for headers."""
    try:
        # Find header line
        header_line_index = None
        for idx, line in enumerate(lines):
            if line.strip().startswith('@'):
                header_line_index = idx
                break
        
        if header_line_index is None:
            logger.error("No header line found in FORAGE.OUT")
            return None
        
        # Fix header: manually handle 'RUN FILEX' if present
        raw_columns = lines[header_line_index].strip().lstrip('@').split()
        if len(raw_columns) >= 2 and raw_columns[0] == "RUN" and raw_columns[1] == "FILEX":
            columns = ["RUN_FILEX"] + raw_columns[2:]
        else:
            columns = raw_columns
        
        # Extract data lines (skip comments)
        data_lines = [line for line in lines[header_line_index + 1:] 
                      if line.strip() and not line.startswith('*')]
        data_text = '\n'.join(data_lines)
        data_io = StringIO(data_text)
        
        # Read data with pandas
        df = pd.read_csv(data_io, delim_whitespace=True, names=columns)
        
        # Standardize treatment column name if needed
        treatment_cols = ["TRNO", "TR", "TRT", "TN"]
        for col in df.columns:
            if col in treatment_cols and col != "TRNO":
                df = df.rename(columns={col: "TRNO"})
                break
        
        # If no treatment column was found, check if first column might be treatments
        if not any(col in df.columns for col in treatment_cols):
            first_col = df.columns[0]
            # Check if first column contains numeric values that could be treatment numbers
            if pd.to_numeric(df[first_col], errors='coerce').notna().all():
                df = df.rename(columns={first_col: "TRNO"})
                logger.info(f"Using column {first_col} as TRNO")
        
        # Convert TRNO to string if it exists
        if "TRNO" in df.columns:
            df["TRNO"] = df["TRNO"].astype(str)
        else:
            # Create default TRNO if none exists
            df["TRNO"] = "1"
            logger.info("Created default TRNO column with value 1")
        
        # Convert other columns to numeric where possible
        for col in df.columns:
            if col != "TRNO":  # Keep TRNO as string
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Drop empty columns
        df = df.loc[:, df.notna().any()]
        
        # Create DATE column if possible
        if "YEAR" in df.columns and "DOY" in df.columns:
            df["DATE"] = pd.to_datetime(
                df["YEAR"].astype(str) + 
                df["DOY"].astype(str).str.zfill(3),
                format="%Y%j",
                errors='coerce'
            )
            df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing forage file: {str(e)}")
        logger.exception("Detailed error:")
        return None

def process_standard_file(lines: List[str]) -> Optional[DataFrame]:
    """Process standard DSSAT output files."""
    try:
        # Process data more efficiently
        data_frames = []
        treatment_indices = [i for i, line in enumerate(lines) if line.strip().upper().startswith("TREATMENT")]

        if treatment_indices:
            # Multiple treatment format
            for idx, start_idx in enumerate(treatment_indices):
                next_idx = treatment_indices[idx + 1] if idx + 1 < len(treatment_indices) else len(lines)
                df = process_treatment_block(lines[start_idx:next_idx])
                if df is not None:
                    data_frames.append(df)
        else:
            # Single treatment format
            df = process_treatment_block(lines)
            if df is not None:
                data_frames.append(df)

        # Combine and process data efficiently
        if data_frames:
            combined_data = pd.concat(data_frames, ignore_index=True)
            combined_data = combined_data.loc[:, combined_data.notna().any()]
            combined_data = standardize_dtypes(combined_data)
            
            # Create DATE column if possible
            if "YEAR" in combined_data.columns and "DOY" in combined_data.columns:
                combined_data["DATE"] = pd.to_datetime(
                    combined_data["YEAR"].astype(str) + 
                    combined_data["DOY"].astype(str).str.zfill(3),
                    format="%Y%j",
                    errors='coerce'
                )
                
            return combined_data

        return None
    
    except Exception as e:
        logger.error(f"Error processing standard file: {str(e)}")
        return None

def process_treatment_block(lines: List[str]) -> Optional[pd.DataFrame]:
    """Process a treatment block of data"""
    try:
        header_index = next((i for i, line in enumerate(lines) if line.startswith("@")), None)
        if header_index is None:
            return None

        headers = lines[header_index].lstrip("@").strip().split()
        data_lines = []
        
        for line in lines[header_index + 1:]:
            line_strip = line.strip()
            # Skip empty lines, comment lines, and metadata lines
            if not line_strip or line_strip.startswith("*"):
                continue
                
            # Skip metadata lines by checking first word
            if len(line_strip.split()) > 0:
                first_word = line_strip.split()[0].upper()
                if first_word in ["MODEL", "EXPERIMENT", "DATA", "!"]:
                    continue
            
            data_lines.append(line_strip.split())

        if not data_lines:
            return None

        df = pd.DataFrame(data_lines, columns=headers)
        
        # Extract treatment number if present
        treatment_line = next((line for line in lines if line.strip().upper().startswith("TREATMENT")), None)
        if treatment_line:
            try:
                trt_num = treatment_line.split()[1]
                df["TRT"] = trt_num
            except IndexError:
                pass

        return df

    except Exception as e:
        logger.error(f"Error processing treatment block: {e}")
        return None


def create_batch_file(input_data: dict, DSSAT_BASE: str) -> str:
    """Create DSSAT batch file for treatment execution."""
    try:
        # Validate input
        required_fields = ["folders", "executables", "experiment", "treatment"]
        missing_fields = [field for field in required_fields if not input_data.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required input data: {', '.join(missing_fields)}")
            
        # Get crop directory
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details 
             if crop['name'].upper() == input_data["folders"].upper()),
            None
        )
        
        if not crop_info:
            raise ValueError(f"Could not find crop information for {input_data['folders']}")
            
        # Process treatments
        treatments = input_data["treatment"]
        if isinstance(treatments, str):
            treatments = [treatments]
            
        # Setup paths
        base_path = os.path.normpath(DSSAT_BASE)
        folder_path = crop_info['directory'].strip()
        
        if not folder_path:
            raise ValueError(f"No directory found for crop {input_data['folders']}")
            
        folder_path = os.path.normpath(folder_path)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
            
        # Create batch file content
        batch_file_lines = [
            f"$BATCH({crop_info['code']})",
            "!",
            f"! Directory    : {folder_path}",
            f"! Command Line : {os.path.join(base_path, input_data['executables'])} B BatchFile.v48",
            f"! Experiment   : {input_data['experiment']}",
            f"! ExpNo        : {len(treatments)}",
            "!",
            "@FILEX                                                                                        TRTNO     RP     SQ     OP     CO"
        ]
        
        # Add treatment lines
        for treatment in treatments:
            try:
                trt_num = int(treatment)
                full_path = os.path.normpath(os.path.join(folder_path, input_data["experiment"]))
                
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Experiment file does not exist: {full_path}")
                    
                padded_path = f"{full_path:<90}"
                line = f"{padded_path}{trt_num:>9}      1      0      0      0"
                batch_file_lines.append(line)
                
            except ValueError as e:
                raise ValueError(f"Invalid treatment number: {treatment}")
                
        # Write batch file
        batch_file_path = os.path.join(folder_path, "BatchFile.v48")
        with open(batch_file_path, "w", newline="\n", encoding='utf-8') as f:
            f.write("\n".join(batch_file_lines))
            
        logger.info(f"Created batch file: {batch_file_path}")
        return batch_file_path
        
    except Exception as e:
        logger.error(f"Error creating batch file: {str(e)}")
        raise

def run_treatment(input_data: dict, DSSAT_BASE: str) -> str:
    """Run DSSAT treatment."""
    if not input_data.get("treatment"):
        raise ValueError("No treatments selected")
        
    try:
        # Get crop directory
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details 
             if crop['name'].upper() == input_data["folders"].upper()),
            None
        )
        
        if not crop_info:
            raise ValueError(f"Could not find crop information for {input_data['folders']}")
            
        # Setup working directory
        work_dir = crop_info['directory'].strip()
        if not os.path.exists(work_dir):
            raise FileNotFoundError(f"Working directory does not exist: {work_dir}")
            
        # Change to working directory
        original_dir = os.getcwd()
        os.chdir(work_dir)
        logger.info(f"Working in directory: {work_dir}")
        
        try:
            # Verify executable and batch file
            exe_path = os.path.normpath(os.path.join(DSSAT_BASE, input_data["executables"]))
            if not os.path.exists(exe_path):
                raise FileNotFoundError(f"Executable not found: {exe_path}")
                
            if not os.path.exists("BatchFile.v48"):
                raise FileNotFoundError("BatchFile.v48 not found in working directory")
                
            # Run DSSAT
            cmd = f'"{exe_path}" B BatchFile.v48'
            logger.info(f"Executing: {cmd}")
            
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                encoding='utf-8'
            )
            
            # Handle execution results
            if result.returncode == 99:
                error_msg = (
                    "DSSAT simulation failed. Please verify:\n"
                    "1. Input files are properly formatted\n"
                    "2. All required weather files are present\n"
                    "3. Cultivation and treatment parameters are valid"
                )
                raise RuntimeError(error_msg)
            elif result.returncode != 0:
                error_msg = result.stderr or f"Unknown error (code {result.returncode})"
                raise RuntimeError(f"DSSAT execution failed: {error_msg}")
                
            return result.stdout
            
        finally:
            os.chdir(original_dir)
            logger.info("Restored original directory")
            
    except Exception as e:
        logger.error(f"Error in run_treatment: {str(e)}")
        raise

def read_evaluate_file(selected_folder: str) -> Optional[DataFrame]:
    """Read and process EVALUATE.OUT file."""
    try:
        # Get crop details
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details 
              if crop['name'].upper() == selected_folder.upper()),
            None
        )
        
        if not crop_info:
            logger.error(f"Could not find crop information for folder {selected_folder}")
            return None
            
        # Construct file path
        folder_path = crop_info['directory'].strip()
        evaluate_path = os.path.join(folder_path, "EVALUATE.OUT")
        
        if not os.path.exists(evaluate_path):
            logger.warning(f"EVALUATE.OUT not found in {folder_path}")
            return None
            
        # Read file
        try:
            with open(evaluate_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(evaluate_path, 'r', encoding='latin-1') as file:
                lines = file.readlines()
                
        # Find header
        header_idx = next(
            (i for i, line in enumerate(lines) 
              if line.strip().startswith("@")),
            None
        )
        
        if header_idx is None:
            logger.error(f"No header found in {evaluate_path}")
            return None
            
        # Process data
        headers = lines[header_idx].strip().lstrip("@").split()
        logger.info(f"Found headers: {headers}")
        
        data_lines = [
            line.strip().split()
            for line in lines[header_idx + 1:]
            if line.strip() and not line.startswith("*")
        ]
        
        if not data_lines:
            logger.warning(f"No data found in {evaluate_path}")
            return None
            
        # Create DataFrame
        df = DataFrame(data_lines, columns=headers)
        logger.info(f"Initial DataFrame columns: {df.columns.tolist()}")
        
        # Convert string columns to numeric where possible
        for col in df.columns:
            df[col] = to_numeric(df[col], errors='coerce')
        
        # Handle missing values - replace with NaN
        for val in config.MISSING_VALUES:
            df = df.replace(val, nan)
        
        # Standardize treatment column names with case-insensitive check
        treatment_cols = ['TRNO', 'TR', 'TRT','TN']
        found_trt_col = None
        for col in df.columns:
            if col.upper() in [t.upper() for t in treatment_cols]:
                found_trt_col = col
                if col != 'TRNO':
                    df = df.rename(columns={col: 'TRNO'})
                    logger.info(f"Renamed '{col}' column to 'TRNO'")
                break
                
        if found_trt_col is None:
            logger.warning("No treatment column (TRNO/TR/TRT) found in the data")
            # Create a default TRNO column if none exists
            df['TRNO'] = 1
            logger.info("Created default TRNO column with value 1")
            
        # Log final columns for debugging
        logger.info(f"Final DataFrame columns: {df.columns.tolist()}")
        
        df = standardize_dtypes(df)
        return df
        
    except Exception as e:
        logger.error(f"Error reading EVALUATE.OUT: {str(e)}")
        logger.exception("Detailed error:")
        return None
    




def read_observed_data(selected_folder: str, selected_experiment: str, x_var: str, y_vars: List[str]) -> Optional[DataFrame]:
    """Read observed data from .xxT file matching experiment name pattern."""
    try:
        base_name = selected_experiment.split(".")[0]
        logger.info(f"Looking for observed data for experiment: {base_name}")
        
        # Get crop details
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details 
             if crop['name'].upper() == selected_folder.upper()),
            None
        )
        
        if not crop_info:
            logger.error(f"Could not find crop code for folder {selected_folder}")
            return None
            
        # Use crop directory
        folder_path = crop_info['directory'].strip()
        logger.info(f"Checking for T file in folder: {folder_path}")
        
        # Look for T file
        t_file_pattern = os.path.join(folder_path, f"{base_name}.{crop_info['code']}T")
        logger.info(f"Looking for files matching pattern: {t_file_pattern}")
        matching_files = [f for f in glob.glob(t_file_pattern) 
                         if not f.upper().endswith(".OUT")]
        
        if not matching_files:
            logger.error(f"No matching .{crop_info['code']}T files found for {base_name}")
            return None
        
        logger.info(f"Found matching T file: {matching_files[0]}")    
        # Read and process T file
        t_file = matching_files[0]
        with open(t_file, "r") as file:
            content = file.readlines()
        
        logger.info(f"Read {len(content)} lines from {t_file}")
        logger.info(f"First 5 lines of file: {content[:5]}")
            
        # Find all header lines
        header_indices = [i for i, line in enumerate(content) if line.strip().startswith("@")]
        
        if not header_indices:
            logger.error(f"No header line found in {t_file}")
            return None
        
        logger.info(f"Found {len(header_indices)} header lines at positions: {header_indices}")
        
        # Process all header lines and combine them
        all_headers = []
        header_sections = []  # Group headers by sections
        
        for idx in header_indices:
            line_headers = content[idx].strip().lstrip("@").split()
            line_headers = [h.upper() for h in line_headers]
            logger.info(f"Headers from line {idx}: {line_headers}")
            
            # Store this as a separate header section
            header_sections.append((idx, line_headers))
            all_headers.extend(line_headers)
        
        # Remove duplicates while preserving order
        headers = []
        for h in all_headers:
            if h not in headers:
                headers.append(h)
        
        logger.info(f"Combined headers from all header lines: {headers}")
        
        # Process each section separately, then combine at the end
        all_data_frames = []
        
        for section_idx, (header_line_idx, section_headers) in enumerate(header_sections):
            logger.info(f"Processing section {section_idx+1} with headers: {section_headers}")
            
            # Determine the end of this section (next header or end of file)
            if section_idx < len(header_sections) - 1:
                section_end = header_sections[section_idx + 1][0]
            else:
                section_end = len(content)
            
            # Find data lines for this section
            section_data_lines = content[header_line_idx+1:section_end]
            section_data_lines = [line for line in section_data_lines if line.strip() and 
                                  not line.strip().startswith("*") and 
                                  not line.strip().startswith("@") and
                                  not line.strip().startswith("!")]
            
            if not section_data_lines:
                logger.info(f"No data found for section {section_idx+1}")
                continue
                
            logger.info(f"Found {len(section_data_lines)} data lines for section {section_idx+1}")
            
            # Check if this is a fixed-width format section by analyzing first data line
            first_data_line = section_data_lines[0]
            field_positions = []
            
            # Try to detect positions by finding transitions from space to non-space
            current_pos = 0
            in_field = False
            for i, char in enumerate(first_data_line):
                if not in_field and char != ' ':
                    # Start of a field
                    field_positions.append(i)
                    in_field = True
                elif in_field and char == ' ':
                    # End of a field
                    in_field = False
            
            # Add end position for the last field
            if len(first_data_line.strip()) > 0:
                field_positions.append(len(first_data_line))
            
            logger.info(f"Detected field positions for section {section_idx+1}: {field_positions}")
            
            # Process data rows for this section
            section_data_rows = []
            
            # Decide parsing method based on data characteristics
            is_fixed_width = len(field_positions) > 1 and len(field_positions) <= len(section_headers)
            
            if is_fixed_width:
                logger.info(f"Using fixed-width parsing for section {section_idx+1}")
                
                # Adjust field positions if needed
                if len(field_positions) < len(section_headers):
                    logger.warning(f"Fewer field positions ({len(field_positions)}) than headers ({len(section_headers)})")
                    # Try to estimate positions for remaining fields
                    avg_width = 5
                    while len(field_positions) < len(section_headers):
                        next_pos = field_positions[-1] + avg_width
                        field_positions.append(next_pos)
                
                # Parse fixed-width data
                for line in section_data_lines:
                    if len(line.strip()) == 0:
                        continue
                        
                    padded_line = line.ljust(field_positions[-1] + 5)  # Add padding
                    row_values = []
                    
                    for i in range(len(section_headers)):
                        start_pos = field_positions[i] if i < len(field_positions) else 0
                        end_pos = field_positions[i+1] if i+1 < len(field_positions) else len(padded_line)
                        
                        field_str = padded_line[start_pos:end_pos].strip()
                        row_values.append(field_str if field_str else None)
                    
                    section_data_rows.append(row_values)
            else:
                logger.info(f"Using split-based parsing for section {section_idx+1}")
                
                # Regular space-delimited data
                for line in section_data_lines:
                    line_parts = line.strip().split()
                    
                    # Ensure right number of columns
                    if len(line_parts) < len(section_headers):
                        line_parts.extend([None] * (len(section_headers) - len(line_parts)))
                    elif len(line_parts) > len(section_headers):
                        line_parts = line_parts[:len(section_headers)]
                        
                    section_data_rows.append(line_parts)
            
            if not section_data_rows:
                logger.warning(f"No data rows parsed for section {section_idx+1}")
                continue
                
            logger.info(f"Parsed {len(section_data_rows)} data rows for section {section_idx+1}")
            
            # Create DataFrame for this section
            section_df = DataFrame(section_data_rows, columns=section_headers)
            
            # Check if any data was loaded
            if section_df.empty:
                logger.warning(f"Empty DataFrame for section {section_idx+1}")
                continue
                
            # Add this section's data to the collection
            all_data_frames.append(section_df)
        
        # Check if we found any data
        if not all_data_frames:
            logger.error("No data found in any section")
            return None
            
        # Combine section DataFrames
        # First, check if they all have TRNO and DATE columns for joining
        common_keys = ["TRNO", "DATE"]
        if all(set(common_keys).issubset(df.columns) for df in all_data_frames):
            logger.info("Joining sections on TRNO and DATE")
            
            # Start with the first DataFrame
            result_df = all_data_frames[0]
            
            # Join with each additional DataFrame
            for i, df in enumerate(all_data_frames[1:], 1):
                logger.info(f"Joining with section {i+1} DataFrame")
                result_df = result_df.merge(df, on=common_keys, how='outer')
        else:
            # If we can't join properly, just use the DataFrame with the most rows
            logger.warning("Cannot join sections on common keys, using section with most rows")
            result_df = max(all_data_frames, key=len)
            
        logger.info(f"Combined DataFrame shape: {result_df.shape}")
        
        # Standardize column names
        if "TRNO" in result_df.columns:
            logger.info("Renaming TRNO to TRT")
            result_df = result_df.rename(columns={"TRNO": "TRT"})
        if "TR" in result_df.columns:
            logger.info("Renaming TR to TRT")
            result_df = result_df.rename(columns={"TR": "TRT"})
        if "TN" in result_df.columns:
            logger.info("Renaming TN to TRT")
            result_df = result_df.rename(columns={"TN": "TRT"})
        
        # Remove columns that are all NaN
        logger.info(f"Columns before filtering: {result_df.columns.tolist()}")
        result_df = result_df.loc[:, result_df.notna().any()]
        logger.info(f"Columns after filtering: {result_df.columns.tolist()}")
        
        # Standardize data types
        logger.info("Standardizing data types")
        result_df = standardize_dtypes(result_df)
        
        # Process DATE column
        if "DATE" in result_df.columns:
            logger.info("Processing DATE column")
            
            # Make sure values are properly formatted before conversion
            def clean_date(x):
                if x is None:
                    return None
                x_str = str(x).strip()
                # Check if this is a 5-digit date format (common in agricultural data)
                if x_str.isdigit() and len(x_str) == 5:
                    return x_str
                # If it's just a number, make sure it's properly padded
                if x_str.isdigit():
                    return x_str.zfill(5)
                return x_str
                
            result_df["DATE"] = result_df["DATE"].apply(clean_date)
            result_df["DATE"] = result_df["DATE"].apply(
                lambda x: unified_date_convert(date_str=str(x)) if x is not None else None
            )
            
            original_length = len(result_df)
            result_df = result_df.dropna(subset=["DATE"])
            if len(result_df) < original_length:
                logger.warning(f"Dropped {original_length - len(result_df)} rows with missing DATE values")
            
            if not result_df.empty and isinstance(result_df["DATE"].iloc[0], pd.Timestamp):
                result_df["DATE"] = result_df["DATE"].dt.strftime("%Y-%m-%d")
            
        # Process treatment columns
        for col in ["TRNO", "TRT", "TR", "TN"]:
            if col in result_df.columns:
                logger.info(f"Converting {col} to string")
                result_df[col] = result_df[col].astype(str)
                
        # Validate required variables
        required_vars = ["TRT"] + [var for var in y_vars if var in result_df.columns]
        missing_vars = [var for var in required_vars if var not in result_df.columns]
        
        if missing_vars:
            logger.warning(f"Missing required variables: {missing_vars}")
            
        # Final check on data size
        logger.info(f"Final DataFrame shape: {result_df.shape}")
        if result_df.empty:
            logger.error("DataFrame is empty after processing")
            return None
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error reading observed data: {str(e)}")
        logger.exception("Detailed traceback:")
        return None