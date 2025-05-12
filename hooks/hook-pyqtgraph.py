
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all pyqtgraph submodules as hidden imports
hiddenimports = collect_submodules('pyqtgraph')

# Collect any data files
datas = collect_data_files('pyqtgraph')
