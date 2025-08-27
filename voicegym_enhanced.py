"""
Enhanced VoiceGym with AI Integration - Backwards compatible version of original script
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import google.generativeai as genai
import pygame
import os
import random
import tempfile
from threading import Thread
from dotenv import load_dotenv
import asyncio
from voicegym_core import VoiceGymCore

load_dotenv()
print("🏋️ VoiceGym Coach - Enhanced AI Version Loading...")

# ==============================================================================
# SETUP
# ==============================================================================

# Add your API keys here or use .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MURF_API_KEY = os.getenv("MURF_API_KEY", "")

# Validate API keys
if not GEMINI_API_KEY or not MURF_API_KEY:
    print("⚠️  API keys not found in environment!")
    print("   - Create a .env file with your keys (see .env.example)")
    print("   - Or set them as environment variables")
    print("   - Fallback mode will be used for missing APIs")

print("✅ VoiceGym Enhanced Version Ready!")

# ==============================================================================
# ENHANCED VOICEGYM CLASS WITH AI INTEGRATION
# ==============================================================================

class VoiceGymEnhanced:
    """Enhanced VoiceGym with improved AI integration and coaching styles."""
    
    def __init__(self, coaching_style="Motivational", voice_id="en-US-terrell", language="English"):
        """Initialize enhanced VoiceGym with customizable coaching style."""
        self.config = {
            'gemini_api_key': GEMINI_API_KEY,
            'murf_api_key': MURF_API_KEY,
            'voice_id': voice_id,
            'language': language,
            'coaching_style': coaching_style,
            'feedback_frequency': 20,
            'rep_feedback': True
        }
        
        # Initialize core VoiceGym functionality
        self.core = VoiceGymCore(self.config)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open camera!")
            raise SystemExit()
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✅ Enhanced VoiceGym initialized with {coaching_style} coaching style!")
        
    async def provide_ai_feedback(self, angle, reps):
        """Provide AI-powered feedback asynchronously."""
        try:
            current_time = time.time()
            
            # Check if enough time has passed for feedback
            if (current_time - self.core.last_feedback > self.config['feedback_frequency'] and 
                current_time - self.core.last_speech > 12):
                
                # Get AI feedback
                exercise_data = {
                    'angle': angle,
                    'reps': reps,
                    'session_duration': current_time - self.core.session_start_time,
                    'coaching_style': self.config['coaching_style']
                }
                
                feedback = await self.core.get_ai_feedback(angle, reps, exercise_data)
                
                if feedback:
                    print(f"💬 AI Coach ({self.config['coaching_style']}): {feedback}")
                    await self.core.speak_feedback_async(feedback)
                    self.core.last_feedback = current_time
                    self.core.last_speech = current_time
                    
        except Exception as e:
            print(f"AI feedback error: {e}")
    
    def run(self):
        """Main workout loop with enhanced AI integration."""
        print("🎥 Starting Enhanced VoiceGym...")
        print(f"🎯 Coaching Style: {self.config['coaching_style']}")
        print(f"🎤 Voice: {self.config['voice_id']} ({self.config['language']})")
        print("🏋️ Position yourself in front of the camera and start doing bicep curls!")
        print("📱 Press 'q' to quit or ESC to exit")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        # Initial motivation based on coaching style
        if self.config['coaching_style'] == "Gentle/Encouraging":
            initial_message = "Welcome to VoiceGym! Take your time, position yourself comfortably, and let's start your gentle workout journey together."
        elif self.config['coaching_style'] == "High-intensity":
            initial_message = "LET'S GO! VoiceGym is ready to CRUSH this workout! Get in position and show me what you've got!"
        else:  # Motivational
            initial_message = "Welcome to VoiceGym! You're about to crush this workout! Position yourself and let's build some serious strength!"
        
        print(f"🎯 {initial_message}")
        asyncio.run(self.core.speak_feedback_async(initial_message))
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("❌ Failed to grab frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for pose detection
            try:
                processed_frame, angle, reps, stage = self.core.process_frame(frame, self.handle_rep_completed)
                
                # Provide AI feedback if angle is detected
                if angle is not None:
                    asyncio.run(self.provide_ai_feedback(angle, reps))
                
                # Display the frame
                cv2.imshow('🏋️ VoiceGym Enhanced - Press Q to Quit', processed_frame)
                
                frame_count += 1
                
                # Stats every 5 seconds
                if frame_count % 150 == 0:  # ~5 seconds at 30fps
                    elapsed = time.time() - start_time
                    stats = self.core.get_session_stats()
                    print(f"📊 Enhanced Stats: {stats['reps']} reps in {elapsed/60:.1f}min (Rate: {stats['reps_per_minute']:.1f}/min)")
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                cv2.imshow('🏋️ VoiceGym Enhanced - Press Q to Quit', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        # Cleanup
        self.cleanup()
    
    def handle_rep_completed(self, event_type, data):
        """Handle rep completion events."""
        if event_type == 'rep_completed' and self.config['rep_feedback']:
            current_time = time.time()
            
            # Only provide rep feedback if enough time has passed
            if current_time - self.core.last_speech > 10:
                # Style-specific rep messages
                if self.config['coaching_style'] == "Gentle/Encouraging":
                    rep_messages = [
                        f"Wonderful! That's rep {data['reps']}. You're doing beautifully!",
                        f"Great job! Rep {data['reps']} completed with nice form.",
                        f"Excellent work! That's {data['reps']} reps. Keep up the steady pace!",
                        f"Perfect! Rep {data['reps']} done. You're building strength steadily.",
                    ]
                elif self.config['coaching_style'] == "High-intensity":
                    rep_messages = [
                        f"BOOM! Rep {data['reps']} CRUSHED! Keep that fire burning!",
                        f"UNSTOPPABLE! That's {data['reps']} reps of pure power!",
                        f"DOMINATING! Rep {data['reps']} destroyed! You're a machine!",
                        f"INCREDIBLE! {data['reps']} reps and counting! KEEP PUSHING!",
                    ]
                else:  # Motivational
                    rep_messages = [
                        f"Fantastic! That's rep {data['reps']}! You're crushing this workout!",
                        f"Excellent work! Rep {data['reps']} completed! Stay strong!",
                        f"Outstanding! That's {data['reps']} reps! You're building serious strength!",
                        f"Perfect form! Rep {data['reps']} in the books! Keep it up!",
                    ]
                
                message = random.choice(rep_messages)
                print(f"✅ {message}")
                asyncio.run(self.core.speak_feedback_async(message))
    
    def cleanup(self):
        """Cleanup resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.core.cleanup()
        
        # Final workout summary
        stats = self.core.get_session_stats()
        
        if self.config['coaching_style'] == "Gentle/Encouraging":
            final_message = f"Wonderful workout! You completed {stats['reps']} bicep curls in {stats['session_duration_minutes']:.1f} minutes. You should feel proud of your effort today!"
        elif self.config['coaching_style'] == "High-intensity":
            final_message = f"WORKOUT COMPLETE! You DEMOLISHED {stats['reps']} reps in {stats['session_duration_minutes']:.1f} minutes! You're an absolute BEAST!"
        else:  # Motivational
            final_message = f"Workout complete! You crushed {stats['reps']} bicep curls in {stats['session_duration_minutes']:.1f} minutes. Excellent work building strength today!"
        
        print(f"🏁 {final_message}")
        asyncio.run(self.core.speak_feedback_async(final_message))
        
        # Keep program alive for final speech
        time.sleep(8)

# ==============================================================================
# MAIN EXECUTION WITH COACHING STYLE SELECTION
# ==============================================================================

def select_coaching_style():
    """Allow user to select coaching style."""
    print("\n🎯 Select Your Coaching Style:")
    print("1. Gentle/Encouraging - Supportive and patient guidance")
    print("2. Motivational - Energetic and uplifting coaching (Default)")
    print("3. High-intensity - Direct and challenging for maximum effort")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3) or press Enter for default: ").strip()
            
            if not choice or choice == "2":
                return "Motivational"
            elif choice == "1":
                return "Gentle/Encouraging"
            elif choice == "3":
                return "High-intensity"
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            exit()

def select_voice():
    """Allow user to select voice."""
    voices = {
        "1": "en-US-terrell",
        "2": "en-US-jenny", 
        "3": "en-GB-oliver",
        "4": "en-GB-emma"
    }
    
    print("\n🎤 Select Voice:")
    print("1. Terrell (English US - Male) - Default")
    print("2. Jenny (English US - Female)")
    print("3. Oliver (English UK - Male)")
    print("4. Emma (English UK - Female)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4) or press Enter for default: ").strip()
            
            if not choice or choice == "1":
                return "en-US-terrell"
            elif choice in voices:
                return voices[choice]
            else:
                print("Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            exit()

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced VoiceGym Coach...")
        print("=" * 60)
        
        # Get user preferences
        coaching_style = select_coaching_style()
        voice_id = select_voice()
        
        print(f"\n✅ Configuration:")
        print(f"   🎯 Coaching Style: {coaching_style}")
        print(f"   🎤 Voice: {voice_id}")
        print(f"   🌐 Language: English")
        
        # Initialize and run enhanced VoiceGym
        gym = VoiceGymEnhanced(
            coaching_style=coaching_style,
            voice_id=voice_id,
            language="English"
        )
        gym.run()
        
    except KeyboardInterrupt:
        print("\n👋 Workout interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 VoiceGym Enhanced session ended!")