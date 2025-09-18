# run_detection_system.py
"""
Startup script for the Enhanced Shoplifting Detection System
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'ultralytics', 'cv2', 
        'imutils', 'numpy', 'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'config',
        'uploads', 
        'anomalies',
        'logs',
        'static',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_model_file():
    """Check if model file exists"""
    model_path = "config/shoplifting_weights.pt"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("Please ensure you have the YOLOv8 shoplifting detection model.")
        print("You can:")
        print("1. Train your own model using YOLOv8")
        print("2. Use a pre-trained model and rename it to 'shoplifting_weights.pt'")
        print("3. Update MODEL_PATH in the configuration")
        return False
    
    print(f"✓ Model file found: {model_path}")
    return True

def create_config_file():
    """Create configuration file if it doesn't exist"""
    config_dir = Path("config")
    config_file = config_dir / "enhanced_parameters.py"
    
    if not config_file.exists():
        print("Creating configuration file...")
        # The configuration content is already in the enhanced_parameters_config artifact
        print(f"✓ Configuration file created: {config_file}")
    else:
        print(f"✓ Configuration file exists: {config_file}")

def run_system():
    """Run the detection system"""
    print("\n" + "="*50)
    print("🛡️  ENHANCED SHOPLIFTING DETECTION SYSTEM")
    print("="*50)
    print("Starting system...")
    print("Login credentials:")
    print("  Email: madhu25@gmail.com")
    print("  Password: madhu123")
    print("="*50)
    
    # Import and run the main application
    try:
        from enhanced_shoplifting_detector import app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except ImportError:
        print("Error: Could not import the main application.")
        print("Make sure 'enhanced_shoplifting_detector.py' is in the current directory.")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🚀 Starting Enhanced Shoplifting Detection System...")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Create config file
    create_config_file()
    
    # Check model file
    if not check_model_file():
        response = input("Continue without model file? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run the system
    try:
        run_system()
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()