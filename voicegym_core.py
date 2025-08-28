"""
VoiceGym Core Module
===================

Core functionality shared between web and CLI versions of VoiceGym.
Provides camera handling, pose detection, voice synthesis, and exercise tracking.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import pygame
import os
import random
import tempfile
import logging
from threading import Thread
from typing import Optional, Tuple, Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceGymCore:
    """Core VoiceGym functionality for pose detection and exercise tracking."""
    
    def __init__(self):
        """Initialize the VoiceGym core system."""
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise tracking
        self.reps = 0
        self.stage = "ready"
        self.last_feedback = 0
        self.last_rep = 0
        self.last_speech = 0
        self.session_start_time = time.time()
        
        # Configuration
        self.voice_cooldown = 10  # seconds between voice feedback
        self.coaching_interval = 20  # seconds between coaching tips
        
        # Initialize audio system
        self._init_audio()
        
        logger.info("VoiceGym Core initialized successfully")
    
    def _init_audio(self):
        """Initialize pygame mixer for audio playback."""
        try:
            pygame.mixer.init()
            logger.info("Audio system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
    
    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculate angle between 3 points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle
    
    def process_pose(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process frame for pose detection and exercise tracking.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, pose_data)
        """
        pose_data = {
            'landmarks_detected': False,
            'angle': 0,
            'stage': self.stage,
            'reps': self.reps,
            'feedback_needed': False,
            'coaching_needed': False
        }
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            pose_data['landmarks_detected'] = True
            
            # Draw pose landmarks
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255,0,255), thickness=2)
            )
            
            # Get arm landmarks for bicep curl detection
            landmarks = results.pose_landmarks.landmark
            try:
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
                
                angle = self.calculate_angle(shoulder, elbow, wrist)
                pose_data['angle'] = angle
                
                # Exercise tracking logic
                current_time = time.time()
                
                # Rep counting
                if angle > 160 and self.stage != "down":
                    self.stage = "down"
                elif angle < 50 and self.stage == "down" and current_time - self.last_rep > 2.0:
                    self.stage = "up"
                    self.reps += 1
                    self.last_rep = current_time
                    
                    # Check if voice feedback is needed
                    if current_time - self.last_speech > self.voice_cooldown:
                        pose_data['feedback_needed'] = True
                        self.last_speech = current_time
                
                # Coaching feedback
                if (current_time - self.last_feedback > self.coaching_interval and 
                    current_time - self.last_speech > 12):
                    pose_data['coaching_needed'] = True
                    self.last_feedback = current_time
                    self.last_speech = current_time
                
                pose_data['stage'] = self.stage
                pose_data['reps'] = self.reps
                
            except Exception as e:
                logger.error(f"Pose processing error: {e}")
        
        return frame, pose_data
    
    def add_overlay(self, frame: np.ndarray, pose_data: Dict[str, Any]) -> np.ndarray:
        """Add information overlay to the frame."""
        h, w = frame.shape[:2]
        
        # Black background for text
        cv2.rectangle(frame, (10, 10), (min(500, w-10), 120), (0,0,0), -1)
        
        # Main info
        cv2.putText(frame, '🏋️ VoiceGym - Bicep Curls', 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        if pose_data['landmarks_detected']:
            cv2.putText(frame, f"Count: {pose_data['reps']} | Angle: {pose_data['angle']:.0f}° | Stage: {pose_data['stage']}", 
                       (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            # Voice cooldown indicator
            current_time = time.time()
            time_since_speech = current_time - self.last_speech
            if time_since_speech < self.voice_cooldown:
                cv2.putText(frame, f'🔊 Voice cooldown: {self.voice_cooldown-time_since_speech:.1f}s', 
                           (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,100), 2)
        else:
            cv2.putText(frame, 'Position yourself in view for pose detection', 
                       (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        return frame
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        elapsed_time = time.time() - self.session_start_time
        return {
            'reps': self.reps,
            'elapsed_time': elapsed_time,
            'elapsed_minutes': elapsed_time / 60,
            'reps_per_minute': self.reps / (elapsed_time / 60) if elapsed_time > 0 else 0,
            'current_stage': self.stage
        }
    
    def reset_session(self):
        """Reset session statistics."""
        self.reps = 0
        self.stage = "ready"
        self.last_feedback = 0
        self.last_rep = 0
        self.last_speech = 0
        self.session_start_time = time.time()
        logger.info("Session reset")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            pygame.mixer.quit()
            logger.info("Audio system cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class CameraManager:
    """Manages camera initialization and frame capture."""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera manager.
        
        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera with error handling."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera at index {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Test frame capture
            ret, _ = self.cap.read()
            if not ret:
                raise Exception("Cannot read frames from camera")
            
            logger.info(f"Camera initialized successfully: {self.width}x{self.height}")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self.cap = None
            raise
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
        
        return ret, frame
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")


class VoiceFeedbackManager:
    """Manages voice synthesis and audio playback with fallback options."""
    
    def __init__(self):
        """Initialize voice feedback manager."""
        self.murf_api_key = os.getenv("MURF_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.use_murf = self.murf_api_key and self.murf_api_key != "YOUR_MURF_API_KEY_HERE"
        
        # Initialize audio system
        try:
            pygame.mixer.init()
            logger.info("Voice feedback manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
    
    def speak(self, text: str, use_fallback: bool = True) -> bool:
        """
        Generate and play speech from text.
        
        Args:
            text: Text to speak
            use_fallback: Whether to use gTTS fallback if Murf fails
            
        Returns:
            True if speech was successful, False otherwise
        """
        logger.info(f"Speaking: {text[:50]}...")
        
        # Try Murf API first if available
        if self.use_murf:
            if self._speak_murf(text):
                return True
            elif not use_fallback:
                return False
        
        # Fallback to gTTS
        if use_fallback:
            return self._speak_gtts(text)
        
        return False
    
    def _speak_murf(self, text: str) -> bool:
        """Generate speech using Murf AI API."""
        try:
            payload = {
                "text": text,
                "voiceId": "en-US-terrell",
                "format": "MP3",
                "model": "GEN2",
                "returnAsBase64": False
            }
            
            headers = {
                "api-key": self.murf_api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.murf.ai/v1/speech/generate",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'audioFile' in response_data:
                    audio_url = response_data['audioFile']
                    audio_response = requests.get(audio_url, timeout=15)
                    
                    if audio_response.status_code == 200:
                        # Save and play audio
                        temp_dir = tempfile.gettempdir()
                        audio_filename = os.path.join(temp_dir, f"voicegym_{int(time.time())}.mp3")
                        
                        with open(audio_filename, "wb") as f:
                            f.write(audio_response.content)
                        
                        self._play_audio_async(audio_filename)
                        return True
            
            logger.warning(f"Murf API failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Murf API error: {e}")
            return False
    
    def _speak_gtts(self, text: str) -> bool:
        """Generate speech using gTTS as fallback."""
        try:
            from gtts import gTTS
            
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            audio_filename = os.path.join(temp_dir, f"voicegym_gtts_{int(time.time())}.mp3")
            
            tts.save(audio_filename)
            self._play_audio_async(audio_filename)
            
            logger.info("Used gTTS for speech synthesis")
            return True
            
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return False
    
    def _play_audio_async(self, filename: str):
        """Play audio file asynchronously."""
        def play():
            try:
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                
                # Wait for playback to finish, then clean up
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Clean up temporary file
                try:
                    os.remove(filename)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
        
        thread = Thread(target=play)
        thread.daemon = True
        thread.start()


class CoachingEngine:
    """Provides coaching feedback and motivational messages."""
    
    def __init__(self):
        """Initialize coaching engine."""
        self.coaching_styles = {
            'motivational': self._get_motivational_messages,
            'technical': self._get_technical_messages,
            'encouraging': self._get_encouraging_messages
        }
        self.current_style = 'motivational'
    
    def set_coaching_style(self, style: str):
        """Set the coaching style."""
        if style in self.coaching_styles:
            self.current_style = style
            logger.info(f"Coaching style set to: {style}")
        else:
            logger.warning(f"Unknown coaching style: {style}")
    
    def get_rep_message(self, rep_count: int) -> str:
        """Get a message for completing a rep."""
        messages = [
            f"Fantastic! That's rep number {rep_count}! Your form is looking strong and controlled.",
            f"Excellent work! Rep {rep_count} completed! Keep that steady rhythm and controlled movement.",
            f"Great job! That's {rep_count} reps down! You're really building strength with each repetition.",
            f"Perfect! Rep {rep_count} in the books! Your biceps are getting a fantastic workout right now.",
            f"Outstanding! That's rep {rep_count}! Keep focusing on that controlled movement and proper form."
        ]
        return random.choice(messages)
    
    def get_coaching_tip(self, angle: float, rep_count: int) -> str:
        """Get coaching feedback based on arm angle and progress."""
        return self.coaching_styles[self.current_style](angle, rep_count)
    
    def _get_motivational_messages(self, angle: float, rep_count: int) -> str:
        """Get motivational coaching messages."""
        if angle < 30:
            messages = [
                "Incredible contraction! You're really squeezing those biceps at the top. Keep that controlled movement going!",
                "Excellent squeeze at the peak! This is where the real muscle building happens. You're doing amazing!",
                "Perfect form at the top! Your biceps are fully engaged right now. Keep pushing through!",
                "Outstanding peak contraction! You're targeting those muscle fibers perfectly. Stay strong!"
            ]
        elif angle > 170:
            messages = [
                "Great extension! You've got excellent range of motion. Now power through that curl!",
                "Perfect starting position! Your arm is fully extended. Squeeze those biceps hard on the way up!",
                "Excellent stretch! This full range of motion is key for muscle development. Keep it up!",
                "Beautiful extension! You're maximizing your range of motion. Now bring that power!"
            ]
        else:
            messages = [
                "You're in the power zone! This is where maximum muscle activation happens. Keep pushing!",
                "Perfect mid-range position! Your biceps are working their hardest right now. Stay strong!",
                "Excellent technique! You're right in the sweet spot for bicep development. Keep going!",
                "Outstanding form! You're building serious strength with every movement. Don't give up!"
            ]
        
        return random.choice(messages)
    
    def _get_technical_messages(self, angle: float, rep_count: int) -> str:
        """Get technical coaching messages."""
        if angle < 30:
            messages = [
                "Excellent peak contraction at approximately 30 degrees. Focus on the controlled eccentric phase.",
                "Perfect bicep activation in the shortened position. Maintain tension through the lowering phase.",
                "Ideal contraction angle achieved. Remember to control the negative for maximum muscle development.",
                "Optimal peak position. Now focus on a 3-second controlled descent for best results."
            ]
        elif angle > 170:
            messages = [
                "Perfect lengthened position at full extension. Initiate the concentric phase with controlled power.",
                "Excellent range of motion achieved. Begin the curl with steady controlled movement.",
                "Ideal starting position established. Focus on bicep activation throughout the full range.",
                "Full extension completed. Engage your biceps and maintain elbow stability during the curl."
            ]
        else:
            messages = [
                "You're in the mid-range where peak force production occurs. Maintain steady tempo.",
                "Optimal working angle for muscle fiber recruitment. Focus on smooth controlled movement.",
                "Perfect position for maximum muscle activation. Keep the movement steady and controlled.",
                "Ideal angle for strength development. Maintain consistent tempo throughout the range."
            ]
        
        return random.choice(messages)
    
    def _get_encouraging_messages(self, angle: float, rep_count: int) -> str:
        """Get encouraging coaching messages."""
        if angle < 30:
            messages = [
                "Beautiful work! You're really getting the most out of each rep. Keep that great form!",
                "Wonderful contraction! I can see the improvement in your technique. You're doing great!",
                "Fantastic squeeze! Your dedication to proper form is really paying off. Keep it up!",
                "Excellent work! You're building strength and improving with every single rep!"
            ]
        elif angle > 170:
            messages = [
                "Great job on that full extension! Your range of motion is looking fantastic!",
                "Perfect form! You're really taking care to do each rep properly. That's awesome!",
                "Wonderful technique! Your commitment to quality movement is inspiring. Keep going!",
                "Excellent work! You're showing great attention to detail in your form. Well done!"
            ]
        else:
            messages = [
                "You're doing wonderfully! Your form is looking better with each rep. Keep it up!",
                "Great work! I can see your strength and technique improving. You're doing amazing!",
                "Fantastic job! Your dedication to proper form is really showing. Keep going strong!",
                "Excellent progress! You're building both strength and skill with every movement!"
            ]
        
        return random.choice(messages)
    
    def get_session_summary(self, stats: Dict[str, Any]) -> str:
        """Get a session summary message."""
        reps = stats['reps']
        minutes = stats['elapsed_minutes']
        
        summary_messages = [
            f"Workout complete! You did {reps} bicep curls in {minutes:.1f} minutes. Great job building strength today!",
            f"Fantastic session! {reps} quality reps in {minutes:.1f} minutes. Your dedication is paying off!",
            f"Excellent work! {reps} controlled reps completed in {minutes:.1f} minutes. You're getting stronger!",
            f"Outstanding effort! {reps} bicep curls in {minutes:.1f} minutes. Keep up the amazing work!"
        ]
        
        return random.choice(summary_messages)


# Utility functions
def get_api_keys() -> Dict[str, Optional[str]]:
    """Get API keys from environment variables."""
    return {
        'murf': os.getenv("MURF_API_KEY"),
        'gemini': os.getenv("GEMINI_API_KEY")
    }

def validate_api_keys() -> Dict[str, bool]:
    """Validate that API keys are properly configured."""
    keys = get_api_keys()
    validation = {}
    
    for service, key in keys.items():
        validation[service] = (
            key is not None and 
            key != f"YOUR_{service.upper()}_API_KEY_HERE" and
            len(key.strip()) > 0
        )
    
    return validation