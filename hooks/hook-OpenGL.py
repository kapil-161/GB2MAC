
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all PyOpenGL submodules
hiddenimports = collect_submodules('OpenGL')

# Don't forget the OpenGL_accelerate module
hiddenimports += collect_submodules('OpenGL_accelerate')

# Collect any data files
datas = collect_data_files('OpenGL')
