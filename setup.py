#!/usr/bin/env python3
"""
VoiceGym Setup Script
====================

Quick setup script to help users get VoiceGym running easily.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible. Need Python 3.8+")
        return False

def check_dependencies():
    """Check if required system dependencies are available."""
    dependencies = {
        'pip': 'pip --version',
        'git': 'git --version'
    }
    
    print("\n🔍 Checking system dependencies...")
    all_good = True
    
    for dep, cmd in dependencies.items():
        if shutil.which(dep):
            print(f"✅ {dep} is available")
        else:
            print(f"❌ {dep} is not available - please install it first")
            all_good = False
    
    return all_good

def setup_environment():
    """Set up the environment file."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        print("\n📝 Setting up environment file...")
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("💡 Edit .env to add your API keys (optional)")
        return True
    elif env_file.exists():
        print("✅ Environment file already exists")
        return True
    else:
        print("❌ .env.example not found")
        return False

def install_requirements():
    """Install Python requirements."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    print("\n📦 Installing Python packages...")
    cmd = f"{sys.executable} -m pip install -r requirements.txt"
    return run_command(cmd, "Installing requirements")

def test_installation():
    """Test if the installation works."""
    print("\n🧪 Testing installation...")
    
    # Test core imports
    try:
        from voicegym_core import VoiceGymCore
        print("✅ Core module imports successfully")
    except ImportError as e:
        print(f"❌ Core module import failed: {e}")
        return False
    
    # Test if structure validation passes
    if Path("test_structure.py").exists():
        result = subprocess.run([sys.executable, "test_structure.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Structure validation passed")
        else:
            print(f"❌ Structure validation failed: {result.stderr}")
            return False
    
    return True

def main():
    """Main setup function."""
    print("🏋️ VoiceGym Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system dependencies
    if not check_dependencies():
        print("\n💡 Please install missing dependencies and run setup again")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed")
        sys.exit(1)
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 VoiceGym setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Edit .env file to add API keys (optional)")
    print("   2. Run web app: streamlit run app.py")
    print("   3. Or run CLI: python voicegym_enhanced.py")
    print("   4. For help: python voicegym_enhanced.py --help")
    print("\n💡 Tip: The app works without API keys using gTTS and pre-scripted coaching!")

if __name__ == "__main__":
    main()