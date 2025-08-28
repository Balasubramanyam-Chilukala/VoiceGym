#!/usr/bin/env python3
"""
VoiceGym Enhanced CLI
====================

Enhanced command-line version of VoiceGym with improved error handling,
multiple coaching styles, and gTTS fallback for voice feedback.
"""

import cv2
import time
import argparse
import logging
import signal
import sys
from typing import Optional

from voicegym_core import (
    VoiceGymCore, 
    CameraManager, 
    VoiceFeedbackManager, 
    CoachingEngine,
    validate_api_keys
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('voicegym.log')
    ]
)
logger = logging.getLogger(__name__)


class VoiceGymEnhanced:
    """Enhanced VoiceGym CLI application with improved features."""
    
    def __init__(self, coaching_style: str = 'motivational', camera_index: int = 0):
        """
        Initialize VoiceGym Enhanced.
        
        Args:
            coaching_style: Style of coaching ('motivational', 'technical', 'encouraging')
            camera_index: Camera device index
        """
        self.running = False
        
        # Initialize core components
        try:
            logger.info("Initializing VoiceGym Enhanced...")
            
            self.core = VoiceGymCore()
            self.camera = CameraManager(camera_index=camera_index)
            self.voice = VoiceFeedbackManager()
            self.coach = CoachingEngine()
            
            # Set coaching style
            self.coach.set_coaching_style(coaching_style)
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("VoiceGym Enhanced initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VoiceGym Enhanced: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def print_startup_info(self):
        """Print startup information and API key status."""
        print("🏋️" + "=" * 58 + "🏋️")
        print("           🎯 VoiceGym Enhanced - Your AI Trainer")
        print("🏋️" + "=" * 58 + "🏋️")
        print()
        
        # Check API key status
        api_validation = validate_api_keys()
        
        print("📋 System Status:")
        print(f"   🎥 Camera: {'✅ Ready' if self.camera.is_opened() else '❌ Not available'}")
        print(f"   🔊 Audio: ✅ Ready")
        print(f"   🗣️  Murf API: {'✅ Available' if api_validation['murf'] else '❌ Not configured (will use gTTS)'}")
        print(f"   🤖 Gemini API: {'✅ Available' if api_validation['gemini'] else '❌ Not configured'}")
        print(f"   👨‍🏫 Coaching Style: {self.coach.current_style.title()}")
        print()
        
        print("📖 Instructions:")
        print("   • Position yourself in front of the camera")
        print("   • Start doing bicep curls with your LEFT arm")
        print("   • The system will track your reps and provide feedback")
        print("   • Press 'q' or ESC to quit")
        print("   • Press 'r' to reset session statistics")
        print("   • Press 's' to show current statistics")
        print()
        
        if not api_validation['murf']:
            print("💡 Note: Murf API not configured. Using gTTS for voice feedback.")
            print("   Set MURF_API_KEY in your .env file for premium voice synthesis.")
            print()
    
    def run_workout(self) -> bool:
        """
        Run the main workout loop.
        
        Returns:
            True if workout completed successfully, False if error occurred
        """
        if not self.camera.is_opened():
            logger.error("Camera not available")
            return False
        
        self.print_startup_info()
        
        # Welcome message
        welcome_msg = (
            f"Welcome to VoiceGym Enhanced! I'm your {self.coach.current_style} trainer. "
            "Position yourself in front of the camera and start your bicep curl workout. "
            "I'll guide you through proper form and count your reps!"
        )
        print(f"🎯 {welcome_msg}")
        logger.info("Starting workout session")
        
        # Speak welcome message
        self.voice.speak(welcome_msg)
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                # Read frame from camera
                ret, frame = self.camera.read_frame()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process pose and get exercise data
                processed_frame, pose_data = self.core.process_pose(frame)
                
                # Add information overlay
                processed_frame = self.core.add_overlay(processed_frame, pose_data)
                
                # Handle voice feedback
                if pose_data['feedback_needed']:
                    rep_message = self.coach.get_rep_message(pose_data['reps'])
                    print(f"✅ {rep_message}")
                    self.voice.speak(rep_message)
                
                # Handle coaching feedback
                if pose_data['coaching_needed']:
                    coaching_tip = self.coach.get_coaching_tip(pose_data['angle'], pose_data['reps'])
                    print(f"💬 Coach: {coaching_tip}")
                    self.voice.speak(coaching_tip)
                
                # Display frame
                cv2.imshow('🏋️ VoiceGym Enhanced - Press Q to Quit', processed_frame)
                
                frame_count += 1
                
                # Print stats every 5 seconds
                if frame_count % 150 == 0:  # ~5 seconds at 30fps
                    stats = self.core.get_session_stats()
                    logger.info(f"Workout stats: {stats['reps']} reps in {stats['elapsed_minutes']:.1f} minutes")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Quit requested by user")
                    break
                elif key == ord('r'):  # Reset session
                    self.reset_session()
                elif key == ord('s'):  # Show stats
                    self.show_stats()
            
            # Workout complete
            stats = self.core.get_session_stats()
            summary = self.coach.get_session_summary(stats)
            print(f"\n🏁 {summary}")
            logger.info(f"Workout completed: {stats}")
            
            # Speak summary
            self.voice.speak(summary)
            
            # Keep program alive for final speech
            time.sleep(3)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during workout: {e}")
            print(f"❌ An error occurred during the workout: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def reset_session(self):
        """Reset the current workout session."""
        self.core.reset_session()
        print("\n🔄 Session reset! Starting fresh...")
        logger.info("Session reset by user")
    
    def show_stats(self):
        """Display current session statistics."""
        stats = self.core.get_session_stats()
        print(f"\n📊 Current Stats:")
        print(f"   Reps: {stats['reps']}")
        print(f"   Time: {stats['elapsed_minutes']:.1f} minutes")
        print(f"   Rate: {stats['reps_per_minute']:.1f} reps/minute")
        print(f"   Stage: {stats['current_stage']}")
        print()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.running = False
            self.camera.release()
            cv2.destroyAllWindows()
            self.core.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for VoiceGym Enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="VoiceGym Enhanced - AI-Powered Personal Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voicegym_enhanced.py                    # Default motivational coaching
  python voicegym_enhanced.py --style technical # Technical coaching style
  python voicegym_enhanced.py --camera 1        # Use camera device 1
  python voicegym_enhanced.py --style encouraging --camera 0  # Encouraging style with camera 0

Coaching Styles:
  motivational  - High-energy, motivational feedback (default)
  technical     - Technical form analysis and biomechanical feedback
  encouraging   - Supportive and encouraging feedback
        """
    )
    
    parser.add_argument(
        '--style', 
        choices=['motivational', 'technical', 'encouraging'],
        default='motivational',
        help='Coaching style for feedback (default: motivational)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-voice',
        action='store_true',
        help='Disable voice feedback (text only)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and run VoiceGym Enhanced
        voicegym = VoiceGymEnhanced(
            coaching_style=args.style,
            camera_index=args.camera
        )
        
        # Disable voice if requested
        if args.no_voice:
            logger.info("Voice feedback disabled")
            # Create a dummy voice manager that doesn't actually speak
            class DummyVoice:
                def speak(self, text, use_fallback=True):
                    return True
            voicegym.voice = DummyVoice()
        
        success = voicegym.run_workout()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n👋 Workout interrupted by user")
        logger.info("Workout interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()