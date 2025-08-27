"""
VoiceGym Core Module - Modular version of the exercise tracking functionality
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
from typing import Dict, Optional, List, Callable
import asyncio
import aiohttp

class VoiceGymCore:
    """Core VoiceGym functionality for exercise tracking and feedback."""
    
    def __init__(self, config: Dict):
        """Initialize VoiceGym with configuration."""
        self.config = config
        self.setup_apis()
        self.setup_pose_detection()
        self.reset_session()
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
            self.audio_available = True
        except Exception as e:
            print(f"Warning: Audio initialization failed: {e}")
            self.audio_available = False
        
    def setup_apis(self):
        """Setup API configurations."""
        if self.config.get('gemini_api_key'):
            genai.configure(api_key=self.config['gemini_api_key'])
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
            
    def setup_pose_detection(self):
        """Setup MediaPipe pose detection."""
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def reset_session(self):
        """Reset session data."""
        self.reps = 0
        self.stage = "ready"
        self.last_feedback = 0
        self.last_rep = 0
        self.last_speech = 0
        self.session_start_time = time.time()
        self.rep_history = []
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between 3 points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle
    
    async def get_ai_feedback(self, angle: float, reps: int, exercise_data: Dict) -> str:
        """Get AI-powered coaching feedback from Gemini."""
        try:
            if not self.gemini_model:
                return self.get_fallback_feedback(angle, reps)
                
            coaching_style = self.config.get('coaching_style', 'Motivational')
            
            prompt = f"""
            You are a professional fitness coach providing real-time feedback during a bicep curl exercise. 
            
            Current exercise data:
            - Current arm angle: {angle}°
            - Completed reps: {reps}
            - Coaching style preference: {coaching_style}
            
            Exercise context:
            - Angles < 30°: Peak contraction (top of curl)
            - Angles > 170°: Full extension (bottom of curl)  
            - Angles 50-120°: Power zone (mid-range)
            
            Provide a brief, encouraging coaching tip (1-2 sentences) that matches the {coaching_style} style:
            - Gentle/Encouraging: Supportive, patient, positive reinforcement
            - Motivational: Energetic, uplifting, progress-focused
            - High-intensity: Direct, challenging, performance-driven
            
            Focus on form, technique, and motivation. Keep it natural and conversational.
            """
            
            response = await asyncio.to_thread(
                self.gemini_model.generate_content, prompt
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return self.get_fallback_feedback(angle, reps)
                
        except Exception as e:
            print(f"AI feedback error: {e}")
            return self.get_fallback_feedback(angle, reps)
    
    def get_fallback_feedback(self, angle: float, reps: int) -> str:
        """Get fallback coaching feedback when AI is unavailable."""
        coaching_style = self.config.get('coaching_style', 'Motivational')
        
        if angle < 30:
            if coaching_style == "Gentle/Encouraging":
                feedback_options = [
                    "Great contraction! You're doing wonderfully. Take your time with the controlled descent.",
                    "Perfect squeeze at the top! You're building strength beautifully. Keep breathing steadily.",
                    "Excellent form! Your biceps are fully engaged. Focus on that slow, controlled movement."
                ]
            elif coaching_style == "High-intensity":
                feedback_options = [
                    "CRUSH that contraction! Maximum squeeze! Now control that weight down with authority!",
                    "PEAK POWER! You're dominating this rep! Control the descent and prepare for the next attack!",
                    "MAXIMUM CONTRACTION! You're a machine! Control that negative with precision!"
                ]
            else:  # Motivational
                feedback_options = [
                    "Incredible contraction! You're building serious strength. Keep that controlled movement!",
                    "Outstanding peak squeeze! This is where the magic happens. Control that descent!",
                    "Perfect form at the top! Your biceps are firing perfectly. Stay strong!"
                ]
        elif angle > 170:
            if coaching_style == "Gentle/Encouraging":
                feedback_options = [
                    "Nice extension! You have great range of motion. Now curl up with steady control.",
                    "Perfect starting position! You're doing great. Focus on a smooth curl upward.",
                    "Excellent stretch! Take your time and curl up with good form."
                ]
            elif coaching_style == "High-intensity":
                feedback_options = [
                    "FULL EXTENSION! Now EXPLODE upward! Show those biceps who's boss!",
                    "MAXIMUM STRETCH! Time to ATTACK that curl! Power through with intensity!",
                    "PERFECT POSITION! Now DOMINATE this rep! Curl with maximum effort!"
                ]
            else:  # Motivational
                feedback_options = [
                    "Great extension! Excellent range of motion. Power through that curl!",
                    "Perfect starting position! You're crushing it. Squeeze those biceps hard!",
                    "Beautiful extension! Now focus on a powerful controlled curl up!"
                ]
        elif 50 <= angle <= 120:
            if coaching_style == "Gentle/Encouraging":
                feedback_options = [
                    "You're in the power zone! Keep up the steady, controlled movement.",
                    "Perfect mid-range position! You're doing beautifully. Stay focused.",
                    "Great technique! Your biceps are working well. Keep breathing steadily."
                ]
            elif coaching_style == "High-intensity":
                feedback_options = [
                    "POWER ZONE! This is where champions are made! PUSH THROUGH!",
                    "MAXIMUM ACTIVATION! Your biceps are on fire! KEEP PUSHING!",
                    "CRUSH THIS ZONE! You're in the sweet spot! DOMINATE!"
                ]
            else:  # Motivational
                feedback_options = [
                    "You're in the power zone! Maximum muscle activation happening now!",
                    "Perfect mid-range! Your biceps are working their hardest right now!",
                    "Outstanding form! You're in the sweet spot for muscle development!"
                ]
        else:
            if coaching_style == "Gentle/Encouraging":
                feedback_options = [
                    "You're doing wonderful! Focus on smooth, controlled movements.",
                    "Great work! Remember to control both the lifting and lowering phases.",
                    "Excellent progress! Your form is looking good. Keep it up!"
                ]
            elif coaching_style == "High-intensity":
                feedback_options = [
                    "CONTROL THE WEIGHT! Show it who's boss! PERFECT FORM!",
                    "TECHNIQUE ON POINT! You're a force of nature! KEEP CRUSHING!",
                    "FORM PERFECTION! You're unstoppable! MAINTAIN THAT INTENSITY!"
                ]
            else:  # Motivational
                feedback_options = [
                    "Looking fantastic! Control those movements for maximum effectiveness!",
                    "Great work! Keep those elbows stable and focus on smooth movement!",
                    "Excellent technique! You're building serious strength!"
                ]
        
        return random.choice(feedback_options)
    
    async def speak_feedback_async(self, text: str) -> bool:
        """Convert text to speech using Murf API asynchronously."""
        try:
            if not self.config.get('murf_api_key'):
                print(f"🔊 Feedback: {text}")
                return False
                
            payload = {
                "text": text,
                "voiceId": self.config.get('voice_id', 'en-US-terrell'),
                "format": "MP3",
                "model": "GEN2",
                "returnAsBase64": False
            }
            
            # Add language if specified
            language = self.config.get('language')
            if language and language != 'English':
                # This would need to be implemented based on Murf API language support
                payload["language"] = language
                
            headers = {
                "api-key": self.config['murf_api_key'],
                "Content-Type": "application/json"
            }
            
            print(f"🔊 Generating speech: '{text[:50]}...'")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.murf.ai/v1/speech/generate",
                    headers=headers,
                    json=payload,
                    timeout=15
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        audio_length = response_data.get('audioLengthInSeconds', 0)
                        print(f"📊 Audio Length: {audio_length} seconds")
                        
                        if 'audioFile' in response_data:
                            audio_url = response_data['audioFile']
                            print(f"🔗 Downloading audio...")
                            
                            async with session.get(audio_url, timeout=15) as audio_response:
                                if audio_response.status == 200:
                                    audio_data = await audio_response.read()
                                    
                                    # Create temporary file
                                    temp_dir = tempfile.gettempdir()
                                    audio_filename = os.path.join(temp_dir, f"voicegym_{int(time.time())}.mp3")
                                    
                                    with open(audio_filename, "wb") as f:
                                        f.write(audio_data)
                                    
                                    file_size = len(audio_data)
                                    print(f"✅ Audio saved: {file_size} bytes")
                                    
                                    # Play audio asynchronously
                                    self.play_audio_async(audio_filename)
                                    return True
                                else:
                                    print(f"❌ Failed to download audio: {audio_response.status}")
                    else:
                        error_text = await response.text()
                        print(f"❌ Murf API Error {response.status}: {error_text}")
                        
        except Exception as e:
            print(f"💥 Speech error: {e}")
        
        return False
    
    def play_audio_async(self, filename):
        """Play audio in a separate thread to avoid blocking."""
        thread = Thread(target=self.play_audio, args=(filename,))
        thread.daemon = True
        thread.start()
    
    def play_audio(self, filename):
        """Play audio file using pygame."""
        try:
            if not self.audio_available:
                print(f"🔊 Audio playback not available: {filename}")
                return False
                
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            print(f"🎵 Playing audio: {filename}")
            return True
        except Exception as e:
            print(f"Audio playback error: {e}")
            return False
    
    def process_frame(self, frame, feedback_callback: Optional[Callable] = None):
        """Process video frame for pose detection and exercise tracking."""
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
                
                angle = self.calculate_angle(shoulder, elbow, wrist)
                current_time = time.time()
                
                # Count reps
                if angle > 160 and self.stage != "down":
                    self.stage = "down"
                elif angle < 50 and self.stage == "down" and current_time - self.last_rep > 2.0:
                    self.stage = "up"
                    self.reps += 1
                    self.last_rep = current_time
                    
                    # Record rep in history
                    self.rep_history.append({
                        'rep_number': self.reps,
                        'timestamp': current_time,
                        'angle': angle
                    })
                    
                    # Trigger feedback callback if provided
                    if feedback_callback:
                        feedback_callback('rep_completed', {
                            'reps': self.reps,
                            'angle': angle,
                            'timestamp': current_time
                        })
                
                # Add text overlay on video
                h, w = frame.shape[:2]
                
                # Black background for text
                cv2.rectangle(frame, (10, 10), (min(500, w-10), 150), (0,0,0), -1)
                
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
                
                # Session info
                session_time = current_time - self.session_start_time
                cv2.putText(frame, f'⏱️ Session: {session_time/60:.1f}min', 
                           (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 2)
                
                return frame, angle, self.reps, self.stage
                
            except Exception as e:
                print(f"Pose processing error: {e}")
        
        return frame, None, self.reps, self.stage
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        return {
            'reps': self.reps,
            'session_duration_minutes': session_duration / 60,
            'rep_history': self.rep_history,
            'current_stage': self.stage,
            'reps_per_minute': self.reps / (session_duration / 60) if session_duration > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.audio_available:
                pygame.mixer.quit()
        except:
            pass