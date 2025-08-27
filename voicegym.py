
import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import google.generativeai as genai
import pygame
import os
import random
from threading import Thread
import tempfile
from dotenv import load_dotenv
import logging
from gtts import gTTS
load_dotenv()
print("🏋️ VoiceGym Coach - Local Machine Version Loading...")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# SETUP
# ==============================================================================

# Add your API keys here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your actual key
MURF_API_KEY = os.getenv("MURF_API_KEY")      # Replace with your actual key

# Validate API keys
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    logger.warning("⚠️ Gemini API key not configured. AI feedback will be limited.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Gemini API configured successfully")
    except Exception as e:
        logger.error(f"❌ Failed to configure Gemini API: {e}")

if not MURF_API_KEY or MURF_API_KEY == "YOUR_MURF_API_KEY_HERE":
    logger.warning("⚠️ Murf API key not configured. Will use gTTS fallback for voice synthesis.")

# Initialize pygame mixer for audio playback with error handling
AUDIO_AVAILABLE = False
try:
    # Try to initialize pygame mixer
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
    logger.info("✅ Audio system initialized successfully")
except pygame.error as e:
    logger.warning(f"⚠️ Audio device initialization failed: {e}")
    logger.info("🔇 Running in silent mode - voice feedback disabled")
except Exception as e:
    logger.warning(f"⚠️ Unexpected audio initialization error: {e}")
    logger.info("🔇 Running in silent mode - voice feedback disabled")

print("✅ System initialized!")

# ==============================================================================
# CAMERA AND AUDIO FUNCTIONS
# ==============================================================================

def test_audio():
    """Test audio system functionality."""
    if not AUDIO_AVAILABLE:
        logger.warning("🔇 Audio system not available for testing")
        return False
    
    try:
        # Generate a simple test beep
        test_text = "Audio test successful"
        logger.info("🔊 Testing audio with gTTS...")
        
        # Use gTTS for testing
        tts = gTTS(text=test_text, lang='en')
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"audio_test_{int(time.time())}.mp3")
        tts.save(test_file)
        
        # Test playback
        result = play_audio(test_file)
        
        # Cleanup
        try:
            os.remove(test_file)
        except:
            pass
            
        if result:
            logger.info("✅ Audio test completed successfully")
        else:
            logger.warning("⚠️ Audio test failed during playback")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Audio test failed: {e}")
        return False

def play_audio(filename):
    """Play audio file using pygame."""
    if not AUDIO_AVAILABLE:
        logger.warning("🔇 Audio playback skipped - no audio device available")
        return False
        
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        logger.info(f"🎵 Playing audio: {os.path.basename(filename)}")
        return True
    except Exception as e:
        logger.error(f"❌ Audio playback error: {e}")
        return False

def play_audio_async(filename):
    """Play audio in a separate thread to avoid blocking."""
    if not AUDIO_AVAILABLE:
        logger.warning("🔇 Audio playback skipped - no audio device available")
        return
        
    thread = Thread(target=play_audio, args=(filename,))
    thread.daemon = True
    thread.start()

# ==============================================================================
# POSE PROCESSING
# ==============================================================================
def calculate_angle(a, b, c):
    """Calculate angle between 3 points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def speak_feedback(text):
    """Text to speech with Murf API primary and gTTS fallback."""
    if not AUDIO_AVAILABLE:
        logger.info(f"🔇 Voice feedback (silent): {text[:50]}...")
        return False
    
    # Try Murf API first if available
    if MURF_API_KEY and MURF_API_KEY != "YOUR_MURF_API_KEY_HERE":
        try:
            logger.info(f"🔊 Generating speech with Murf API: '{text[:50]}...'")
            
            # Correct Murf API payload structure
            payload = {
                "text": text,
                "voiceId": "en-US-terrell",
                "format": "MP3",
                "model": "GEN2",
                "returnAsBase64": False
            }
            
            headers = {
                "api-key": MURF_API_KEY,
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
                audio_length = response_data.get('audioLengthInSeconds', 0)
                logger.info(f"📊 Murf Audio Length: {audio_length} seconds")
                
                if 'audioFile' in response_data:
                    audio_url = response_data['audioFile']
                    logger.info("🔗 Downloading Murf audio...")
                    
                    audio_response = requests.get(audio_url, timeout=15)
                    if audio_response.status_code == 200:
                        # Create temporary file
                        temp_dir = tempfile.gettempdir()
                        audio_filename = os.path.join(temp_dir, f"voicegym_murf_{int(time.time())}.mp3")
                        
                        with open(audio_filename, "wb") as f:
                            f.write(audio_response.content)
                        
                        file_size = len(audio_response.content)
                        logger.info(f"✅ Murf audio saved: {file_size} bytes")
                        
                        # Play audio asynchronously
                        play_audio_async(audio_filename)
                        return True
                    else:
                        logger.warning(f"⚠️ Failed to download Murf audio: {audio_response.status_code}")
                        
            else:
                logger.warning(f"⚠️ Murf API Error {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.warning(f"⚠️ Murf API error: {e}")
    
    # Fallback to gTTS
    try:
        logger.info(f"🔊 Generating speech with gTTS fallback: '{text[:50]}...'")
        
        tts = gTTS(text=text, lang='en')
        temp_dir = tempfile.gettempdir()
        audio_filename = os.path.join(temp_dir, f"voicegym_gtts_{int(time.time())}.mp3")
        
        tts.save(audio_filename)
        logger.info("✅ gTTS audio generated successfully")
        
        # Play audio asynchronously
        play_audio_async(audio_filename)
        return True
        
    except Exception as e:
        logger.error(f"❌ gTTS fallback error: {e}")
    
    logger.error("❌ All voice synthesis methods failed")
    return False

def get_coaching_tip(angle, reps):
    """Get AI coaching feedback with detailed messages."""
    try:
        if angle < 30:
            feedback_options = [
                "Incredible contraction! You're really squeezing those biceps at the top. Keep that controlled movement going and focus on the slow descent.",
                "Excellent squeeze at the peak! This is where the real muscle building happens. Control that negative movement for maximum gains.",
                "Perfect form at the top! Your biceps are fully engaged right now. Remember to breathe and control the weight down slowly.",
                "Outstanding peak contraction! You're targeting those muscle fibers perfectly. Keep that controlled tempo throughout the entire movement."
            ]
        elif angle > 170:
            feedback_options = [
                "Great extension! You've got excellent range of motion. Now focus on a powerful but controlled curl up, engaging your core.",
                "Perfect starting position! Your arm is fully extended. Squeeze those biceps hard as you bring the weight up with control.",
                "Excellent stretch! This full range of motion is key for muscle development. Keep those elbows stable as you curl up.",
                "Beautiful extension! You're maximizing your range of motion. Now power through that curl with steady controlled movement."
            ]
        elif 50 <= angle <= 120:
            feedback_options = [
                "You're in the power zone! This is where maximum muscle activation happens. Keep pushing through with steady control.",
                "Perfect mid-range position! Your biceps are working their hardest right now. Focus on that smooth controlled movement.",
                "Excellent technique in the working zone! This angle is ideal for muscle fiber recruitment. Stay strong and controlled.",
                "Outstanding form! You're right in the sweet spot for bicep development. Maintain that steady rhythm and breathing pattern."
            ]
        else:
            feedback_options = [
                "Looking fantastic! Remember to control both the lifting and lowering phases for maximum effectiveness and muscle growth.",
                "Great work with your form! Keep those elbows stable, core engaged, and focus on smooth controlled movements throughout.",
                "Solid technique! You're building serious strength with that movement pattern. Keep the weight under control at all times.",
                "Excellent progress! Your form is improving with each rep. Focus on steady breathing and controlled muscle engagement."
            ]
        
        return random.choice(feedback_options)
        
    except Exception as e:
        print(f"AI coaching error: {e}")
        return "Keep pushing! You're doing fantastic! Focus on controlled movements and proper form for the best results and muscle development."

# ==============================================================================
# MAIN GYM CLASS
# ==============================================================================
class VoiceGymLocal:
    def __init__(self, camera_index=0):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.reps = 0
        self.stage = "ready"
        self.last_feedback = 0
        self.last_rep = 0
        self.last_speech = 0
        self.camera_index = camera_index
        self.cap = None
        
        # Initialize camera with better error handling
        self.init_camera()
        
    def init_camera(self):
        """Initialize camera with proper error handling."""
        try:
            logger.info(f"🎥 Initializing camera {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                # Try alternative camera indices
                for alt_index in [1, 2, -1]:
                    logger.warning(f"⚠️ Camera {self.camera_index} failed, trying camera {alt_index}")
                    self.cap = cv2.VideoCapture(alt_index)
                    if self.cap.isOpened():
                        self.camera_index = alt_index
                        break
                        
                if not self.cap.isOpened():
                    logger.error("❌ Cannot open any camera!")
                    raise SystemExit("No camera available")
                    
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test camera by grabbing a frame
            ret, frame = self.cap.read()
            if not ret:
                logger.error("❌ Cannot read from camera!")
                raise SystemExit("Camera not working")
                
            logger.info(f"✅ Camera {self.camera_index} initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Camera initialization failed: {e}")
            raise
    
    def release_camera(self):
        """Properly release camera resources."""
        if self.cap:
            self.cap.release()
            logger.info("📹 Camera released")
    
    def get_available_cameras(self):
        """Get list of available camera indices."""
        available_cameras = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
        
    def process_frame(self, frame):
        """Process video frame for pose detection."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            # Draw pose
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255,0,255), thickness=2)
            )
            
            # Get arm points
            landmarks = results.pose_landmarks.landmark
            try:
                # Using left arm for bicep curl detection
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
                
                angle = calculate_angle(shoulder, elbow, wrist)
                current_time = time.time()
                
                # Count reps
                if angle > 160 and self.stage != "down":
                    self.stage = "down"
                elif angle < 50 and self.stage == "down" and current_time - self.last_rep > 2.0:
                    self.stage = "up"
                    self.reps += 1
                    self.last_rep = current_time
                    
                    # Only speak if enough time has passed since last speech (10 seconds)
                    if current_time - self.last_speech > 10:
                        rep_messages = [
                            f"Fantastic! That's rep number {self.reps}! Your form is looking strong and controlled.",
                            f"Excellent work! Rep {self.reps} completed! Keep that steady rhythm and controlled movement.",
                            f"Great job! That's {self.reps} reps down! You're really building strength with each repetition.",
                            f"Perfect! Rep {self.reps} in the books! Your biceps are getting a fantastic workout right now.",
                            f"Outstanding! That's rep {self.reps}! Keep focusing on that controlled movement and proper form."
                        ]
                        
                        rep_message = random.choice(rep_messages)
                        print(f"✅ {rep_message}")
                        speak_feedback(rep_message)
                        self.last_speech = current_time
                    else:
                        print(f"✅ Rep {self.reps}! (Voice feedback cooling down...)")
                
                # Coaching feedback every 20 seconds (increased gap)
                if (current_time - self.last_feedback > 20 and 
                    current_time - self.last_speech > 12):  # 12 seconds since last speech
                    
                    tip = get_coaching_tip(angle, self.reps)
                    if tip:
                        print(f"💬 Coach: {tip}")
                        speak_feedback(tip)
                        self.last_feedback = current_time
                        self.last_speech = current_time
                
                # Add text overlay on video
                h, w = frame.shape[:2]
                
                # Black background for text
                cv2.rectangle(frame, (10, 10), (min(500, w-10), 120), (0,0,0), -1)
                
                # Main info
                cv2.putText(frame, f'🏋️ VoiceGym - Bicep Curls', 
                           (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f'Count: {self.reps} | Angle: {angle:.0f}° | Stage: {self.stage}', 
                           (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                
                # Timing info
                time_since_speech = current_time - self.last_speech
                if time_since_speech < 10:
                    cv2.putText(frame, f'🔊 Voice cooldown: {10-time_since_speech:.1f}s', 
                               (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,100), 2)
                else:
                    cv2.putText(frame, '', 
                               (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 2)
                
            except Exception as e:
                print(f"Pose processing error: {e}")
        
        return frame
    
    def run(self):
        """Main workout loop."""
        print("🎥 Starting VoiceGym...")
        print("🏋️ Position yourself in front of the camera and start doing bicep curls!")
        print("📱 Press 'q' to quit or ESC to exit")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        # Initial motivation
        initial_message = "Welcome to VoiceGym! Position yourself in front of the camera and start your bicep curl workout. I'll guide you through proper form!"
        print(f"🎯 {initial_message}")
        speak_feedback(initial_message)
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("❌ Failed to grab frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for pose detection
            try:
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow('🏋️ VoiceGym Coach - Press Q to Quit', processed_frame)
                
                frame_count += 1
                
                # Stats every 5 seconds
                if frame_count % 150 == 0:  # ~5 seconds at 30fps
                    elapsed = time.time() - start_time
                    print(f"📊 Workout Stats: {self.reps} reps completed in {elapsed/60:.1f} minutes")
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                cv2.imshow('🏋️ VoiceGym Coach - Press Q to Quit', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        # Cleanup
        self.release_camera()
        cv2.destroyAllWindows()
        if AUDIO_AVAILABLE:
            pygame.mixer.quit()
        
        # Final workout summary
        elapsed = time.time() - start_time
        final_message = f"Workout complete! You did {self.reps} bicep curls in {elapsed/60:.1f} minutes. Great job building strength today!"
        logger.info(f"🏁 {final_message}")
        speak_feedback(final_message)
        
        # Keep program alive for final speech
        time.sleep(8)

def get_coaching_tip(angle, reps):
    """Get detailed coaching feedback based on arm position."""
    try:
        if angle < 30:
            feedback_options = [
                "Incredible contraction! You're really squeezing those biceps at the top. Keep that controlled movement going and focus on the slow descent for maximum muscle activation.",
                "Excellent squeeze at the peak! This is where the real muscle building happens. Control that negative movement slowly for maximum gains and strength development.",
                "Perfect form at the top! Your biceps are fully engaged right now. Remember to breathe steadily and control the weight down slowly for optimal results.",
                "Outstanding peak contraction! You're targeting those muscle fibers perfectly. Keep that controlled tempo throughout the entire movement for best results."
            ]
        elif angle > 170:
            feedback_options = [
                "Great extension! You've got excellent range of motion there. Now focus on a powerful but controlled curl up, engaging your core for stability.",
                "Perfect starting position! Your arm is fully extended beautifully. Squeeze those biceps hard as you bring the weight up with complete control.",
                "Excellent stretch! This full range of motion is absolutely key for muscle development. Keep those elbows stable as you curl up powerfully.",
                "Beautiful extension! You're maximizing your range of motion perfectly. Now power through that curl with steady controlled movement and focus."
            ]
        elif 50 <= angle <= 120:
            feedback_options = [
                "You're in the power zone! This is where maximum muscle activation happens. Keep pushing through with steady control and proper breathing.",
                "Perfect mid-range position! Your biceps are working their hardest right now. Focus on that smooth controlled movement and muscle engagement.",
                "Excellent technique in the working zone! This angle is ideal for muscle fiber recruitment. Stay strong, controlled, and keep that rhythm going.",
                "Outstanding form! You're right in the sweet spot for bicep development. Maintain that steady rhythm and proper breathing pattern throughout."
            ]
        else:
            feedback_options = [
                "Looking fantastic! Remember to control both the lifting and lowering phases for maximum effectiveness and optimal muscle growth.",
                "Great work with your form! Keep those elbows stable, core engaged, and focus on smooth controlled movements throughout each repetition.",
                "Solid technique! You're building serious strength with that movement pattern. Keep the weight under complete control at all times.",
                "Excellent progress! Your form is improving with each rep. Focus on steady breathing and controlled muscle engagement for best results."
            ]
        
        return random.choice(feedback_options)
        
    except Exception as e:
        print(f"AI coaching error: {e}")
        return "Keep pushing! You're doing fantastic! Focus on controlled movements and proper form for the best results and optimal muscle development."

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        print("🚀 Starting VoiceGym Coach...")
        print("=" * 60)
        
        # Test audio system
        if AUDIO_AVAILABLE:
            logger.info("🔊 Testing audio system...")
            test_audio()
        
        gym = VoiceGymLocal()
        gym.run()
        
    except KeyboardInterrupt:
        logger.info("\n👋 Workout interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        if AUDIO_AVAILABLE:
            pygame.mixer.quit()
        logger.info("🏁 VoiceGym Coach session ended!")