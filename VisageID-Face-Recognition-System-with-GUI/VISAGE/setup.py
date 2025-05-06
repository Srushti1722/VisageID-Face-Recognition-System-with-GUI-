from cx_Freeze import setup, Executable
import sys
import os

# Detect Python installation directory and set environment variables
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

# Set base for Windows applications
base = "Win32GUI" if sys.platform == "win32" else None

# Define the target executable
executables = [Executable("train.py", base=base)]

# List of required packages
packages = [
    "os", "sys", "tkinter", "cv2", "numpy", "PIL", "pandas", "datetime", "time"
]

# Include additional files if needed
include_files = [
    os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),
    os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll')
]

# Build options
options = {
    'build_exe': {
        'packages': packages,
        'include_files': include_files,
        'excludes': ["pytest", "unittest"],  # Exclude unnecessary modules
        'optimize': 2  # Enable optimization level 2
    }
}

# Setup script
setup(
    name="VisionToolBox",
    version="1.0.0",
    description="A Vision ToolBox application",
    options=options,
    executables=executables
)
