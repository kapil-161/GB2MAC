
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Add missing dependencies
hiddenimports = [
    'jaraco.text', 
    'plistlib', 
    'appdirs', 
    'packaging', 
    'packaging.version', 
    'packaging.specifiers', 
    'packaging.requirements', 
    'packaging.markers',
    'pkg_resources.extern.packaging',
    'setuptools.extern.packaging',
    'importlib_metadata',
    'zipp',
    'attrs',
    'more_itertools'
]

# Add data files
datas = collect_data_files('pkg_resources')
