#!/usr/bin/env python3
"""
WSGI configuration for Smart Ball Detection System
For PythonAnywhere deployment
"""

import sys
import os

# Add your project directory to the sys.path
path = '/home/yourusername/smart-ball-detection'
if path not in sys.path:
    sys.path.insert(0, path)

# Import your Flask app
from app_flask import app as application

# This is for PythonAnywhere
if __name__ == "__main__":
    application.run()
