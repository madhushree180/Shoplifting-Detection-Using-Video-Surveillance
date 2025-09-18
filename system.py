# install_system.py
"""
Installation script for the Enhanced Shoplifting Detection System
"""
import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "flask==2.3.3",
        "flask-cors==4.0.0", 
        "ultralytics==8.0.200",
        "opencv-python==4.8.1.78",
        "imutils==0.5.4",
        "numpy==1.24.3",
        "Werkzeug==2.3.7",
        "Pillow==10.0.1"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_project_structure():
    """Create the project directory structure"""
    print("📁 Creating project structure...")
    
    directories = [
        "config",
        "uploads",
        "anomalies", 
        "logs",
        "static/css",
        "static/js",
        "static/images",
        "templates",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

def create_sample_files():
    """Create sample configuration and documentation files"""
    print("📄 Creating sample files...")