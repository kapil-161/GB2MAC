
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Make sure we include all attrs submodules
hiddenimports = collect_submodules('attrs')

# Add data files
datas = collect_data_files('attrs')
