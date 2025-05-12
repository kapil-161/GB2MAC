from PyInstaller.utils.hooks import collect_all

# Ensure all ctypes modules are included
datas, binaries, hiddenimports = collect_all('ctypes')

# Add additional specific imports that might be needed
hiddenimports += [
    'ctypes.util',
    'ctypes._endian',
    'ctypes.macholib',
    'ctypes.macholib.dyld',
    'ctypes.macholib.dylib',
    'ctypes.macholib.framework'
]