
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Define required Qt plugins
qt_plugins = [
    ('platforms', ['platforms/libqcocoa.dylib'])
]

# All hidden imports for our app
hidden_imports = [
    # PyQt6 and UI dependencies
    'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtWidgets', 'PyQt6.QtGui', 
    'pyqtgraph',
    
    # Data processing dependencies
    'pandas.core.frame', 'pandas.core.series', 'numpy.core',
    
    # pkg_resources dependencies
    'jaraco.text', 'plistlib', 'appdirs', 
    'packaging', 'packaging.version', 'packaging.specifiers', 
    'packaging.requirements', 'packaging.markers',
    'pkg_resources.extern.packaging', 'setuptools.extern.packaging',
    'importlib_metadata', 'zipp', 'attrs', 'more_itertools',
    
    # OpenGL dependencies
    'OpenGL', 'OpenGL.GL', 'OpenGL.GLU', 'OpenGL.GLUT', 'OpenGL.platform'
]

# Minimal exclusion list - only large/unnecessary packages
exclusions = [
    'matplotlib', 'scipy', 'tkinter', '_tkinter', 'Tkinter', 'wx',
    'IPython', 'notebook', 'sphinx', 'docutils',
    'PIL', 'sqlalchemy', 'tornado', 'jinja2', 'flask',
    'PyQt6.QtWebEngine', 'PyQt6.QtMultimedia', 'PyQt6.QtNetwork',
    'PyQt6.QtQml', 'PyQt6.QtQuick', 
    'plotly', 'dash', 'PyQt5','PySide6',
]

a = Analysis(
    ['main.py'],  # Entry point to your application
    pathex=[],
    binaries=[],
    datas=[('resources', 'resources')],  # Include resources directory
    hiddenimports=hidden_imports,
    hookspath=['hooks'],  # Use our hooks directory
    hooksconfig={'pyqt6': {'plugins': qt_plugins}},
    runtime_hooks=[
        f'hooks/pkg_resources_hook.py',  # pkg_resources patch
        f'hooks/opengl_hook.py',
        f'hooks/ctypes_fix.py'  # OpenGL patch
    ],
    excludes=exclusions,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter unnecessary Qt plugins
def filter_binaries(binaries):
    excluded_patterns = [
        'QtWebEngine', 'QtDesigner', 'QtQuick', 'QtQml', 'QtHelp',
        'QtMultimedia'
    ]
    
    return [(name, path, typ) for name, path, typ in binaries 
            if not any(pattern in name for pattern in excluded_patterns)]

a.binaries = filter_binaries(a.binaries)

# Add a script to create an empty __init__.py file to prevent scanning issues
init_script = '''
import os
import sys

# Create an empty __init__.py in the root directory
base_dir = os.path.dirname(sys.executable)
init_path = os.path.join(base_dir, "__init__.py")

with open(init_path, "w") as f:
    f.write("# Empty __init__.py to prevent scanning issues")
'''

a.scripts.append(('pkg_resources_workaround', init_script, 'PYSOURCE'))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='dssat_viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging, False for production
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None, icon=r'/Applications/DSSAT48/Tools/GBTOOLS/GBuild2/resources/final.png',
)
