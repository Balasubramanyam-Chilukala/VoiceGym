
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
load_dotenv()
print("🏋️ VoiceGym Coach - Local Machine Version Loading...")

# ==============================================================================
# SETUP
# ==============================================================================

# Add your API keys here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your actual key
MURF_API_KEY = os.getenv("MURF_API_KEY")      # Replace with your actual key

# Validate API keys
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or MURF_API_KEY == "YOUR_MURF_API_KEY_HERE":
    print("❌ Please add your actual API keys to the script!")
    print("   - Get Gemini API key from: https://makersuite.google.com/app/apikey")
    print("   - Get Murf API key from: https://murf.ai/api")
    raise SystemExit()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize pygame mixer for audio playback
try:
    pygame.mixer.init()
    print("✅ Audio system initialized!")
except Exception as e:
    print(f"⚠️  Audio initialization failed: {e}")
    print("📢 VoiceGym will continue without audio feedback")

print("✅ API Keys configured!")

# ==============================================================================
# CAMERA AND AUDIO FUNCTIONS
# ==============================================================================

def detect_available_cameras():
    """Detect all available camera devices."""
    available_cameras = []
    print("🔍 Detecting available cameras...")
    
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                camera_info = {
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'working': True
                }
                available_cameras.append(camera_info)
                print(f"📷 Camera {i}: {width}x{height} @ {fps:.1f}fps")
            else:
                print(f"⚠️  Camera {i}: Detected but cannot capture frames")
        cap.release()
    
    if not available_cameras:
        print("❌ No working cameras detected!")
    else:
        print(f"✅ Found {len(available_cameras)} working camera(s)")
    
    return available_cameras

def test_camera(camera_index, duration=3):
    """Test a specific camera for a few seconds."""
    print(f"🧪 Testing camera {camera_index} for {duration} seconds...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    start_time = time.time()
    frame_count = 0
    
    print("Press 'q' to stop test early or wait for automatic completion...")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture frame from camera {camera_index}")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Add test overlay
        cv2.putText(frame, f'Camera {camera_index} Test', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Press Q to stop', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_index} Test - Press Q to Stop', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Camera {camera_index} test completed - captured {frame_count} frames")
    return True

def play_audio(filename):
    """Play audio file using pygame."""
    try:
        if not pygame.mixer.get_init():
            print("🔇 Audio not available - skipping audio playback")
            return False
            
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        print(f"🎵 Playing audio: {filename}")
        return True
    except Exception as e:
        print(f"Audio playback error: {e}")
        return False

def play_audio_async(filename):
    """Play audio in a separate thread to avoid blocking."""
    try:
        thread = Thread(target=play_audio, args=(filename,))
        thread.daemon = True
        thread.start()
    except Exception as e:
        print(f"Async audio error: {e}")

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
    """Text to speech via Murf API."""
    try:
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
        
        print(f"🔊 Generating speech: '{text[:50]}...'")
        
        response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            response_data = response.json()
            audio_length = response_data.get('audioLengthInSeconds', 0)
            print(f"📊 Audio Length: {audio_length} seconds")
            
            if 'audioFile' in response_data:
                audio_url = response_data['audioFile']
                print(f"🔗 Downloading audio...")
                
                audio_response = requests.get(audio_url, timeout=15)
                if audio_response.status_code == 200:
                    # Create temporary file
                    temp_dir = tempfile.gettempdir()
                    audio_filename = os.path.join(temp_dir, f"voicegym_{int(time.time())}.mp3")
                    
                    with open(audio_filename, "wb") as f:
                        f.write(audio_response.content)
                    
                    file_size = len(audio_response.content)
                    print(f"✅ Audio saved: {file_size} bytes")
                    
                    # Play audio asynchronously
                    play_audio_async(audio_filename)
                    return True
                else:
                    print(f"❌ Failed to download audio: {audio_response.status_code}")
                    
        else:
            print(f"❌ Murf API Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"💥 Speech error: {e}")
    
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
    def __init__(self):
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
        
        # Camera will be initialized separately
        self.cap = None
        self.camera_index = None
        self.available_cameras = []
        from threading import Lock
        self._camera_lock = Lock()  # For thread synchronization
        
        print("✅ VoiceGym initialized (camera not yet connected)")
    
    def detect_cameras(self):
        """Detect and return available cameras."""
        self.available_cameras = detect_available_cameras()
        return self.available_cameras
    
    def initialize_camera(self, camera_index=None):
        """Initialize camera with proper error handling and fallbacks."""
        with self._camera_lock:  # Thread synchronization
            if self.cap is not None:
                self.cleanup_camera()
            
            # If no specific camera requested, try to find the best one
            if camera_index is None:
                if not self.available_cameras:
                    self.available_cameras = self.detect_cameras()
                
                if not self.available_cameras:
                    print("❌ No cameras available for initialization!")
                    return False
                
                # Use the first available camera
                camera_index = self.available_cameras[0]['index']
                print(f"🎯 Auto-selecting camera {camera_index}")
            
            print(f"🔧 Initializing camera {camera_index}...")
            
            try:
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    print(f"❌ Cannot open camera {camera_index}")
                    return False
                
                # Test if we can actually capture frames
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    print(f"❌ Camera {camera_index} opened but cannot capture frames")
                    self.cap.release()
                    self.cap = None
                    return False
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Verify settings
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self.camera_index = camera_index
                print(f"✅ Camera {camera_index} initialized successfully!")
                print(f"📐 Resolution: {actual_width}x{actual_height}")
                return True
                
            except Exception as e:
                print(f"❌ Error initializing camera {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False
    
    def try_fallback_cameras(self):
        """Try to initialize any available camera as fallback."""
        print("🔄 Trying fallback cameras...")
        
        if not self.available_cameras:
            self.available_cameras = self.detect_cameras()
        
        for camera_info in self.available_cameras:
            camera_index = camera_info['index']
            if camera_index != self.camera_index:  # Don't retry the same camera
                print(f"🔄 Attempting fallback to camera {camera_index}")
                if self.initialize_camera(camera_index):
                    return True
        
        print("❌ All fallback cameras failed")
        return False
    
    def test_current_camera(self, duration=3):
        """Test the currently initialized camera."""
        if self.cap is None:
            print("❌ No camera initialized for testing")
            return False
        
        return test_camera(self.camera_index, duration)
    
    def cleanup_camera(self):
        """Properly cleanup camera resources."""
        with self._camera_lock:  # Thread synchronization
            if self.cap is not None:
                print(f"🧹 Cleaning up camera {self.camera_index}")
                self.cap.release()
                self.cap = None
                self.camera_index = None
    
    def is_camera_ready(self):
        """Check if camera is ready for use."""
        with self._camera_lock:  # Thread synchronization
            if self.cap is None:
                return False
            
            if not self.cap.isOpened():
                return False
            
            # Quick frame capture test
            ret, frame = self.cap.read()
            return ret and frame is not None
        
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
    
    def setup_camera_interactive(self):
        """Interactive camera setup with user choices."""
        print("\n🎥 CAMERA SETUP")
        print("=" * 50)
        
        # Detect cameras
        cameras = self.detect_cameras()
        
        if not cameras:
            print("❌ No cameras detected! Please check your camera connections.")
            print("💡 Make sure:")
            print("   - Camera is properly connected")
            print("   - Camera drivers are installed")
            print("   - No other applications are using the camera")
            return False
        
        # If only one camera, use it automatically
        if len(cameras) == 1:
            camera_index = cameras[0]['index']
            print(f"📷 Found 1 camera, automatically selecting camera {camera_index}")
            
            if self.initialize_camera(camera_index):
                print("✅ Camera ready!")
                return True
            else:
                print("❌ Failed to initialize the only available camera")
                return False
        
        # Multiple cameras - let user choose
        print(f"📷 Found {len(cameras)} cameras:")
        for i, cam in enumerate(cameras):
            print(f"   {i+1}. Camera {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
        
        while True:
            try:
                choice = input(f"\n🎯 Select camera (1-{len(cameras)}) or 'q' to quit: ").strip().lower()
                
                if choice == 'q':
                    return False
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(cameras):
                    selected_camera = cameras[choice_num - 1]
                    camera_index = selected_camera['index']
                    
                    if self.initialize_camera(camera_index):
                        return True
                    else:
                        print(f"❌ Failed to initialize camera {camera_index}")
                        print("🔄 Please try another camera")
                        continue
                else:
                    print(f"❌ Please enter a number between 1 and {len(cameras)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n👋 Setup cancelled by user")
                return False
    
    def camera_test_menu(self):
        """Camera testing menu."""
        if not self.is_camera_ready():
            print("❌ No camera initialized for testing")
            return False
        
        print(f"\n🧪 CAMERA TEST (Camera {self.camera_index})")
        print("=" * 50)
        
        while True:
            print(f"\nOptions:")
            print("1. Quick test (3 seconds)")
            print("2. Extended test (10 seconds)")
            print("3. Skip test and start workout")
            print("4. Return to camera setup")
            
            try:
                choice = input("Select option (1-4): ").strip()
                
                if choice == '1':
                    if self.test_current_camera(3):
                        print("✅ Quick test passed!")
                        return True
                    else:
                        print("❌ Camera test failed")
                        
                elif choice == '2':
                    if self.test_current_camera(10):
                        print("✅ Extended test passed!")
                        return True
                    else:
                        print("❌ Camera test failed")
                        
                elif choice == '3':
                    print("⏭️  Skipping camera test")
                    return True
                    
                elif choice == '4':
                    return False
                    
                else:
                    print("❌ Please enter 1, 2, 3, or 4")
                    
            except KeyboardInterrupt:
                print("\n👋 Test cancelled by user")
                return False

    def run(self):
        """Main workout loop with improved camera setup."""
        print("🏋️ VoiceGym Coach - Enhanced Version")
        print("=" * 60)
        
        # Interactive camera setup
        while True:
            if self.setup_camera_interactive():
                # Camera initialized successfully
                if self.camera_test_menu():
                    break  # User approved camera, continue to workout
                else:
                    # User wants to change camera
                    self.cleanup_camera()
                    continue
            else:
                # No camera available or user quit
                print("👋 Exiting VoiceGym - camera setup incomplete")
                return
        
        print("\n🎥 Starting VoiceGym workout...")
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
            # Check camera health
            if not self.is_camera_ready():
                print("⚠️  Camera connection lost! Attempting recovery...")
                if not self.try_fallback_cameras():
                    print("❌ Cannot recover camera connection. Ending workout.")
                    break
                print("✅ Camera recovered! Continuing workout...")
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("❌ Failed to grab frame from camera")
                print("🔄 Attempting camera recovery...")
                if not self.try_fallback_cameras():
                    print("❌ Camera recovery failed. Ending workout.")
                    break
                continue
            
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
        self.cleanup_camera()
        cv2.destroyAllWindows()
        try:
            pygame.mixer.quit()
        except:
            pass  # Audio might not be initialized
        
        # Final workout summary
        elapsed = time.time() - start_time
        final_message = f"Workout complete! You did {self.reps} bicep curls in {elapsed/60:.1f} minutes. Great job building strength today!"
        print(f"🏁 {final_message}")
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
    gym = None
    try:
        print("🚀 Starting VoiceGym Coach...")
        print("=" * 60)
        
        gym = VoiceGymLocal()
        gym.run()
        
    except KeyboardInterrupt:
        print("\n👋 Workout interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gym:
            gym.cleanup_camera()
        cv2.destroyAllWindows()
        try:
            pygame.mixer.quit()
        except:
            pass  # Audio might not be initialized
        print("🏁 VoiceGym Coach session ended!")