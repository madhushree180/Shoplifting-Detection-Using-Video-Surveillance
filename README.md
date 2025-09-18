
  readme_content = """# Enhanced Shoplifting Detection System

## Features
- 🔐 Secure login system
- 🎥 Real-time video processing
- 🚨 Automatic anomaly detection and capture
- 📊 Live monitoring dashboard
- 📁 Anomaly image management
- 🔄 Alert system with different severity levels

## Installation
1. Run the installation script: `python install_system.py`
2. Place your YOLOv8 model in `config/shoplifting_weights.pt`
3. Run the system: `python run_detection_system.py`

## Login Credentials
- Email: madhu25@gmail.com
- Password: madhu123

## Usage
1. Access the system at http://localhost:5000
2. Login with the provided credentials
3. Upload a video file
4. Start detection
5. Monitor live feed and captured anomalies

## Directory Structure
- `config/` - Configuration files and model
- `uploads/` - Uploaded video files
- `anomalies/` - Captured anomaly images
- `logs/` - System logs
- `static/` - Static web assets
- `templates/` - HTML templates

## API Endpoints
- `/` - Login page
- `/dashboard` - Main dashboard (requires login)
- `/upload_video` - Upload video file
- `/start_detection` - Start detection
- `/stop_detection` - Stop detection
- `/video_feed` - Live video stream
- `/anomalies` - Get anomaly data
- `/anomaly_images/<filename>` - Serve anomaly images

## Configuration
Edit `config/enhanced_parameters.py` to customize:
- Detection thresholds
- High-value zones
- Behavior patterns
- Alert settings
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✓ Created README.md")

def main():
    """Main installation function"""
    print("🔧 Enhanced Shoplifting Detection System Installer")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Installation failed!")
        sys.exit(1)
    
    # Create project structure
    create_project_structure()
    
    # Create sample files
    create_sample_files()
    
    print("\n✅ Installation completed successfully!")
    print("\nNext steps:")
    print("1. Place your YOLOv8 shoplifting detection model in 'config/shoplifting_weights.pt'")
    print("2. Run: python run_detection_system.py")
    print("3. Open browser to: http://localhost:5000")
    print("4. Login with: madhu25@gmail.com / madhu123")

if __name__ == "__main__":
    main()