#!/usr/bin/env python3
"""
Test script for VoiceGym functionality
"""
import os
import sys
import tempfile
from gtts import gTTS

def test_gtts():
    """Test gTTS functionality without audio playback."""
    try:
        print("🔊 Testing gTTS voice synthesis...")
        
        test_text = "This is a test of Google Text-to-Speech synthesis for VoiceGym"
        tts = gTTS(text=test_text, lang='en')
        
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, "voicegym_test.mp3")
        
        tts.save(test_file)
        
        # Check if file was created and has content
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"✅ gTTS test successful! Generated {file_size} bytes")
            
            # Cleanup
            os.remove(test_file)
            return True
        else:
            print("❌ gTTS test failed - no output file")
            return False
            
    except Exception as e:
        print(f"❌ gTTS test failed: {e}")
        return False

def test_imports():
    """Test all required imports."""
    try:
        print("📦 Testing imports...")
        
        import cv2
        import mediapipe as mp
        import numpy as np
        import requests
        import pygame
        import streamlit
        from gtts import gTTS
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_voicegym_functionality():
    """Test VoiceGym voice synthesis without audio device."""
    try:
        print("🏋️ Testing VoiceGym voice synthesis...")
        
        # Import voicegym module
        sys.path.append('/home/runner/work/VoiceGym/VoiceGym')
        import voicegym
        
        # Test speak_feedback function (should work with gTTS even without audio device)
        # We'll modify it temporarily to not require audio playback
        original_audio_available = voicegym.AUDIO_AVAILABLE
        original_play_audio_async = voicegym.play_audio_async
        
        # Mock the audio playback to just report success
        def mock_play_audio_async(filename):
            print(f"🎵 Mock playback: {os.path.basename(filename)}")
            return True
        
        voicegym.play_audio_async = mock_play_audio_async
        voicegym.AUDIO_AVAILABLE = True  # Temporarily enable audio for testing
        
        # Test the voice synthesis
        result = voicegym.speak_feedback("Testing VoiceGym voice synthesis with gTTS fallback")
        
        # Restore original values
        voicegym.AUDIO_AVAILABLE = original_audio_available
        voicegym.play_audio_async = original_play_audio_async
        
        if result:
            print("✅ VoiceGym voice synthesis test successful!")
            return True
        else:
            print("❌ VoiceGym voice synthesis test failed")
            return False
            
    except Exception as e:
        print(f"❌ VoiceGym test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Running VoiceGym Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("gTTS Synthesis", test_gtts),
        ("VoiceGym Voice", test_voicegym_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed!")
        sys.exit(1)