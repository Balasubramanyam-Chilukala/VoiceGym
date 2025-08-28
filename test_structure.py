#!/usr/bin/env python3
"""
Test VoiceGym Core Structure
============================

Basic test script to verify the structure and imports work correctly.
This script can run without external dependencies to check syntax and structure.
"""

import sys
import os
import importlib.util

def test_module_structure(module_path, module_name):
    """Test if a module has correct structure."""
    print(f"\n🔍 Testing {module_name}...")
    
    if not os.path.exists(module_path):
        print(f"❌ File not found: {module_path}")
        return False
    
    try:
        # Load module spec
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            print(f"❌ Cannot create spec for {module_name}")
            return False
        
        # Read file content for basic syntax check
        with open(module_path, 'r') as f:
            content = f.read()
        
        # Basic checks
        print(f"✅ File exists and readable ({len(content)} chars)")
        
        # Check for key classes/functions
        if module_name == "voicegym_core":
            required_elements = [
                "class VoiceGymCore",
                "class CameraManager", 
                "class VoiceFeedbackManager",
                "class CoachingEngine"
            ]
        elif module_name == "voicegym_enhanced":
            required_elements = [
                "class VoiceGymEnhanced",
                "def main()"
            ]
        elif module_name == "app":
            required_elements = [
                "class VoiceGymApp",
                "def main()"
            ]
        else:
            required_elements = []
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing elements: {missing_elements}")
            return False
        else:
            print(f"✅ All required elements found: {required_elements}")
        
        # Check for basic Python syntax
        try:
            compile(content, module_path, 'exec')
            print("✅ Python syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {module_name}: {e}")
        return False

def test_files_structure():
    """Test the overall file structure."""
    print("🏋️ VoiceGym Core Files Structure Test")
    print("=" * 50)
    
    base_dir = "/home/runner/work/VoiceGym/VoiceGym"
    
    # Expected files
    expected_files = {
        "voicegym_core.py": "Core functionality module",
        "voicegym_enhanced.py": "Enhanced CLI version", 
        "app.py": "Streamlit web application",
        "requirements.txt": "Dependencies file",
        ".env.example": "Environment template",
        "README.md": "Documentation"
    }
    
    print("\n📁 Checking file structure...")
    all_good = True
    
    for filename, description in expected_files.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} (MISSING)")
            all_good = False
    
    # Test module structure
    module_tests = [
        (os.path.join(base_dir, "voicegym_core.py"), "voicegym_core"),
        (os.path.join(base_dir, "voicegym_enhanced.py"), "voicegym_enhanced"),
        (os.path.join(base_dir, "app.py"), "app")
    ]
    
    for module_path, module_name in module_tests:
        if not test_module_structure(module_path, module_name):
            all_good = False
    
    # Check requirements.txt
    print(f"\n📦 Checking requirements.txt...")
    requirements_path = os.path.join(base_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        required_packages = [
            "opencv-python", "mediapipe", "numpy",
            "pygame", "gTTS", "streamlit", "plotly",
            "google-generativeai", "python-dotenv"
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements: {missing_packages}")
            all_good = False
        else:
            print(f"✅ All required packages present in requirements.txt")
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All structure tests passed!")
        print("\n📋 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Copy .env.example to .env and configure API keys")
        print("   3. Test CLI: python voicegym_enhanced.py --help")
        print("   4. Test web app: streamlit run app.py")
    else:
        print("❌ Some structure tests failed. Please review the issues above.")
    
    return all_good

if __name__ == "__main__":
    success = test_files_structure()
    sys.exit(0 if success else 1)