import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import google.generativeai as genai
import pygame
import os
import json
from threading import Thread
import tempfile
from dotenv import load_dotenv

load_dotenv()
print("🏋️ Enhanced VoiceGym Coach - AI Powered Version Loading...")

# ==============================================================================
# SETUP
# ==============================================================================

# Add your API keys here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

# Validate API keys
if not GEMINI_API_KEY or not MURF_API_KEY:
    print("❌ Please add your actual API keys to the .env file!")
    print("   - Get Gemini API key from: https://makersuite.google.com/app/apikey")
    print("   - Get Murf API key from: https://murf.ai/api")
    raise SystemExit()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

pygame.mixer.init()
print("✅ API Keys configured!")

# ==============================================================================
# VOICE AND LANGUAGE CONFIGURATION
# ==============================================================================

def get_supported_voices():
    """Return comprehensive voice configuration with multiple languages."""
    return {
        "English (US)": {
            "en-US": {
                "Male": [
                    {"id": "en-US-ken", "name": "Ken", "style": "Conversational", "description": "American male voice - Conversational style"},
                    {"id": "en-US-carter", "name": "Carter", "style": "Conversational", "description": "American male voice - Multilingual (French supported)"}
                ],
                "Female": [
                    {"id": "en-US-natalie", "name": "Natalie", "style": "Promo", "description": "American female voice - Promotional style"}
                ]
            }
        },
        "English (UK)": {
            "en-UK": {
                "Female": [
                    {"id": "en-UK-ruby", "name": "Ruby", "style": "Conversational", "description": "British female voice - Multilingual (German supported)"}
                ]
            }
        },
        "Hindi (India)": {
            "hi-IN": {
                "Male": [
                    {"id": "hi-IN-amit", "name": "Amit", "style": "General", "description": "Hindi male voice - General style"}
                ],
                "Female": [
                    {"id": "hi-IN-ayushi", "name": "Ayushi", "style": "Conversational", "description": "Hindi female voice - Conversational style"}
                ]
            }
        },
        "Chinese (Mandarin)": {
            "zh-CN": {
                "Male": [
                    {"id": "zh-CN-tao", "name": "Tao", "style": "Conversational", "description": "Chinese male voice - Conversational style"}
                ],
                "Female": [
                    {"id": "zh-CN-jiao", "name": "Jiao", "style": "Conversational", "description": "Chinese female voice - Conversational style"}
                ]
            }
        },
        "French (France)": {
            "fr-FR": {
                "Female": [
                    {"id": "fr-FR-adélie", "name": "Adélie", "style": "Conversational", "description": "French female voice - Conversational style"}
                ]
            }
        },
        "German (Germany)": {
            "de-DE": {
                "Male": [
                    {"id": "de-DE-matthias", "name": "Matthias", "style": "Conversational", "description": "German male voice - Conversational style"}
                ]
            }
        },
        "Spanish (Spain)": {
            "es-ES": {
                "Male": [
                    {"id": "es-ES-javier", "name": "Javier", "style": "Conversational", "description": "Spanish male voice - Conversational style"}
                ],
                "Female": [
                    {"id": "es-ES-elvira", "name": "Elvira", "style": "Conversational", "description": "Spanish female voice - Conversational style"}
                ]
            }
        },
        "Italian (Italy)": {
            "it-IT": {
                "Male": [
                    {"id": "it-IT-lorenzo", "name": "Lorenzo", "style": "Conversational", "description": "Italian male voice - Conversational style"}
                ],
                "Female": [
                    {"id": "it-IT-greta", "name": "Greta", "style": "Conversational", "description": "Italian female voice - Conversational style"}
                ]
            }
        }
    }

def fetch_available_voices():
    """Fetch real voices from Murf API and merge with supported voices."""
    supported_voices = get_supported_voices()
    
    try:
        headers = {
            "api-key": MURF_API_KEY,
            "Content-Type": "application/json"
        }
        
        print("📡 Fetching available voices from Murf API...")
        response = requests.get("https://api.murf.ai/v1/speech/voices", headers=headers, timeout=10)
        
        if response.status_code == 200:
            voices_data = response.json()
            api_voices = {}
            
            # Process API response
            for voice in voices_data.get('voices', []):
                language = voice.get('language', 'Unknown')
                lang_code = voice.get('languageCode', 'unknown')
                gender = voice.get('gender', 'Unknown')
                voice_id = voice.get('voiceId', '')
                name = voice.get('name', voice_id)
                style = voice.get('style', 'General')
                
                # Group by language
                if language not in api_voices:
                    api_voices[language] = {lang_code: {"Male": [], "Female": []}}
                if lang_code not in api_voices[language]:
                    api_voices[language][lang_code] = {"Male": [], "Female": []}
                
                voice_info = {
                    "id": voice_id,
                    "name": name,
                    "style": style,
                    "description": f"{gender} voice in {language} - {style} style"
                }
                
                api_voices[language][lang_code][gender].append(voice_info)
            
            print(f"✅ Found {len(voices_data.get('voices', []))} voices from Murf API")
            
            # Merge API voices with supported voices (prioritize supported ones)
            for lang, lang_codes in supported_voices.items():
                if lang not in api_voices:
                    api_voices[lang] = lang_codes
                else:
                    for code, genders in lang_codes.items():
                        if code not in api_voices[lang]:
                            api_voices[lang][code] = genders
                        else:
                            for gender, voices in genders.items():
                                if gender not in api_voices[lang][code]:
                                    api_voices[lang][code][gender] = voices
                                else:
                                    # Add supported voices that might not be in API response
                                    existing_ids = {v['id'] for v in api_voices[lang][code][gender]}
                                    for voice in voices:
                                        if voice['id'] not in existing_ids:
                                            api_voices[lang][code][gender].append(voice)
            
            return api_voices
            
    except Exception as e:
        print(f"❌ Error fetching voices from API: {e}")
    
    # Fallback to supported voices if API fails
    print("🔄 Using supported voice configuration...")
    return supported_voices

def display_voice_menu(available_voices):
    """Display voice selection menu organized by language and get user choice."""
    print("\n🎤 VOICE SELECTION MENU")
    print("=" * 70)
    
    voice_options = []
    counter = 1
    
    # Sort languages for better display
    language_order = [
        "English (US)", "English (UK)", "Hindi (India)", "Chinese (Mandarin)",
        "French (France)", "German (Germany)", "Spanish (Spain)", "Italian (Italy)"
    ]
    
    # Display languages in preferred order, then any remaining
    displayed_languages = set()
    
    for language in language_order:
        if language in available_voices:
            displayed_languages.add(language)
            print(f"\n🌍 {language}:")
            print("-" * 50)
            
            lang_codes = available_voices[language]
            for lang_code, genders in lang_codes.items():
                for gender, voices in genders.items():
                    if voices:  # Only display if there are voices
                        print(f"  👤 {gender} Voices:")
                        for voice in voices:
                            style_info = f" [{voice.get('style', 'General')}]" if voice.get('style') else ""
                            print(f"     {counter}. {voice['name']}{style_info}")
                            print(f"        └── {voice['description']}")
                            
                            voice_options.append({
                                'language': language,
                                'lang_code': lang_code,
                                'voice_id': voice['id'],
                                'name': voice['name'],
                                'gender': gender,
                                'style': voice.get('style', 'General'),
                                'description': voice['description']
                            })
                            counter += 1
    
    # Display any remaining languages not in the preferred order
    for language, lang_codes in available_voices.items():
        if language not in displayed_languages:
            print(f"\n🌍 {language}:")
            print("-" * 50)
            
            for lang_code, genders in lang_codes.items():
                for gender, voices in genders.items():
                    if voices:
                        print(f"  👤 {gender} Voices:")
                        for voice in voices:
                            style_info = f" [{voice.get('style', 'General')}]" if voice.get('style') else ""
                            print(f"     {counter}. {voice['name']}{style_info}")
                            print(f"        └── {voice['description']}")
                            
                            voice_options.append({
                                'language': language,
                                'lang_code': lang_code,
                                'voice_id': voice['id'],
                                'name': voice['name'],
                                'gender': gender,
                                'style': voice.get('style', 'General'),
                                'description': voice['description']
                            })
                            counter += 1
    
    if not voice_options:
        print("❌ No voices available. Using default configuration.")
        return {
            'language': 'English (US)',
            'lang_code': 'en-US',
            'voice_id': 'en-US-ken',
            'name': 'Ken',
            'gender': 'Male',
            'style': 'Conversational',
            'description': 'Default American male voice'
        }
    
    print(f"\n💫 Enter your choice (1-{len(voice_options)}): ", end="")
    
    while True:
        try:
            choice = int(input())
            if 1 <= choice <= len(voice_options):
                selected_voice = voice_options[choice - 1]
                print(f"\n✅ Selected: {selected_voice['name']} ({selected_voice['language']})")
                print(f"   📝 Style: {selected_voice['style']}")
                print(f"   🎭 Voice ID: {selected_voice['voice_id']}")
                return selected_voice
            else:
                print(f"❌ Please enter a number between 1 and {len(voice_options)}: ", end="")
        except ValueError:
            print(f"❌ Please enter a valid number between 1 and {len(voice_options)}: ", end="")

def get_voice_settings():
    """Get voice and language preferences from user."""
    print("\n🎯 Welcome to Enhanced VoiceGym Coach!")
    print("🌍 Choose your AI fitness coach from multiple languages!")
    print("=" * 60)
    
    # Fetch voices (API + supported voices)
    available_voices = fetch_available_voices()
    
    print(f"\n📊 Available Languages: {len(available_voices)}")
    for lang, codes in available_voices.items():
        voice_count = sum(len(voices) for lang_code in codes.values() for voices in lang_code.values())
        print(f"   🌍 {lang}: {voice_count} voices")
    
    voice_config = display_voice_menu(available_voices)
    
    print(f"\n🎊 Excellent choice! Your AI coach will be {voice_config['name']}")
    print(f"🗣️  Language: {voice_config['language']}")
    print(f"🎭 Style: {voice_config['style']}")
    print("🚀 Starting your multilingual workout session...")
    
    return voice_config

# ==============================================================================
# CAMERA AND AUDIO FUNCTIONS
# ==============================================================================

def play_audio(filename):
    """Play audio file using pygame."""
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        print(f"🎵 Playing audio: {os.path.basename(filename)}")
        return True
    except Exception as e:
        print(f"Audio playback error: {e}")
        return False

def play_audio_async(filename):
    """Play audio in a separate thread to avoid blocking."""
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

def speak_feedback(text, voice_config):
    """Enhanced text to speech via Murf API with comprehensive voice support."""
    try:
        # Build payload with all voice parameters
        payload = {
            "text": text,
            "voiceId": voice_config['voice_id'],
            "format": "MP3",
            "model": "GEN2",
            "returnAsBase64": False,
            "language": voice_config['lang_code']
        }
        
        # Add style if available
        if voice_config.get('style'):
            payload["style"] = voice_config['style']
        
        headers = {
            "api-key": MURF_API_KEY,
            "Content-Type": "application/json"
        }
        
        print(f"🔊 {voice_config['name']} ({voice_config['language']}): '{text[:50]}...'")
        
        response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            response_data = response.json()
            audio_length = response_data.get('audioLengthInSeconds', 0)
            
            if 'audioFile' in response_data:
                audio_url = response_data['audioFile']
                
                audio_response = requests.get(audio_url, timeout=15)
                if audio_response.status_code == 200:
                    temp_dir = tempfile.gettempdir()
                    audio_filename = os.path.join(temp_dir, f"voicegym_{int(time.time())}.mp3")
                    
                    with open(audio_filename, "wb") as f:
                        f.write(audio_response.content)
                    
                    play_audio_async(audio_filename)
                    return True
                else:
                    print(f"❌ Failed to download audio: {audio_response.status_code}")
            else:
                print(f"❌ No audio file in response: {response_data}")
                
        else:
            print(f"❌ Murf API Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"💥 Speech error: {e}")
    
    return False

def get_ai_coaching_feedback(angle, reps, stage, workout_duration, voice_config, ai_quota_exceeded=False):
    """Get dynamic AI coaching feedback with language-specific context."""
    
    # If AI quota exceeded, use smart fallback messages
    if ai_quota_exceeded:
        return get_smart_fallback_coaching(angle, reps, stage, workout_duration, voice_config)
    
    try:
        # Create language-aware context for Gemini
        language_context = {
            'en-US': 'American English with energetic, motivational tone',
            'en-UK': 'British English with professional, encouraging tone',
            'hi-IN': 'Hindi context with respectful, encouraging language',
            'zh-CN': 'Chinese context with respectful, motivational approach',
            'fr-FR': 'French context with elegant, encouraging expression',
            'de-DE': 'German context with precise, motivational language',
            'es-ES': 'Spanish context with warm, encouraging expression',
            'it-IT': 'Italian context with passionate, motivational approach'
        }
        
        lang_style = language_context.get(voice_config['lang_code'], 'English with motivational tone')
        
        context = f"""
        You are an expert fitness coach providing real-time feedback during a bicep curl workout.
        
        Current workout context:
        - Exercise: Bicep Curls
        - Current arm angle: {angle:.1f} degrees
        - Current stage: {stage}
        - Total reps completed: {reps}
        - Workout duration: {workout_duration:.1f} minutes
        - Voice: {voice_config['name']} ({voice_config['language']})
        - Language Style: {lang_style}
        - Voice Style: {voice_config.get('style', 'General')}
        
        Angle interpretation:
        - 0-30°: Full contraction (peak of curl)
        - 30-50°: Strong contraction phase
        - 50-120°: Active lifting/lowering phase
        - 120-160°: Extension phase
        - 160-180°: Full extension (bottom of curl)
        
        Provide a motivational, specific coaching tip (1-2 sentences max) that:
        1. Is encouraging and energetic
        2. Gives specific form advice based on current angle
        3. Motivates continued effort
        4. Sounds natural when spoken aloud
        5. Is appropriate for the selected voice personality and language style
        6. Matches the {voice_config.get('style', 'General')} style
        
        Keep response under 25 words for natural speech flow.
        Respond in English (the text will be converted to the target language via TTS).
        """
        
        response = model.generate_content(context)
        
        if response.text:
            feedback = response.text.strip()
            # Clean up any formatting
            feedback = feedback.replace('*', '').replace('#', '').replace('"', '')
            return feedback
        else:
            return get_smart_fallback_coaching(angle, reps, stage, workout_duration, voice_config)
            
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            print(f"⚠️  AI quota reached. Using smart fallback coaching.")
            return get_smart_fallback_coaching(angle, reps, stage, workout_duration, voice_config)
        else:
            print(f"AI coaching error: {e}")
            return get_smart_fallback_coaching(angle, reps, stage, workout_duration, voice_config)

def get_smart_fallback_coaching(angle, reps, stage, workout_duration, voice_config):
    """Smart fallback coaching messages with language awareness."""
    import random
    
    # Language-specific motivational phrases
    language_phrases = {
        'en-US': {
            'excellent': ['Excellent', 'Outstanding', 'Perfect', 'Great'],
            'keep_going': ['Keep going', 'Stay strong', 'Push forward', 'Keep it up'],
            'control': ['controlled', 'steady', 'smooth', 'precise']
        },
        'en-UK': {
            'excellent': ['Brilliant', 'Excellent', 'Superb', 'Splendid'],
            'keep_going': ['Carry on', 'Well done', 'Keep going', 'Brilliant work'],
            'control': ['controlled', 'steady', 'measured', 'precise']
        },
        'hi-IN': {
            'excellent': ['Excellent', 'Bahut badhiya', 'Perfect', 'Outstanding'],
            'keep_going': ['Keep going', 'Shabash', 'Very good', 'Well done'],
            'control': ['controlled', 'steady', 'smooth', 'perfect']
        }
    }
    
    # Get appropriate phrases for language
    phrases = language_phrases.get(voice_config['lang_code'], language_phrases['en-US'])
    
    # Context-aware fallback messages
    if angle < 30:
        messages = [
            f"{random.choice(phrases['excellent'])} contraction! Hold that squeeze and control the descent.",
            f"{random.choice(phrases['excellent'])} peak position! Now slowly lower with control.",
            f"Great squeeze at the top! Focus on that {random.choice(phrases['control'])} negative.",
            f"{random.choice(phrases['excellent'])} contraction! Feel those biceps working hard."
        ]
    elif angle > 170:
        messages = [
            f"Perfect extension! Now power up with {random.choice(phrases['control'])} strength.",
            f"Great range of motion! Squeeze hard on the way up.",
            f"{random.choice(phrases['excellent'])} stretch position! Keep those elbows stable.",
            f"Beautiful extension! Now curl with focused power."
        ]
    elif 50 <= angle <= 120:
        messages = [
            f"You're in the power zone! Keep that {random.choice(phrases['control'])} control.",
            f"Perfect working angle! Maintain that smooth rhythm.",
            f"Great form in the active zone! Stay {random.choice(phrases['control'])}.",
            f"{random.choice(phrases['excellent'])} technique! This is where strength builds."
        ]
    else:
        # General motivational messages
        base_messages = [
            f"Fantastic form! {random.choice(phrases['keep_going'])} with that {random.choice(phrases['control'])} movement.",
            f"Great work! Focus on smooth, {random.choice(phrases['control'])} reps.",
            f"{random.choice(phrases['excellent'])} technique! You're building real strength.",
            f"Outstanding effort! {random.choice(phrases['keep_going'])} with that steady rhythm."
        ]
        
        # Add rep-specific encouragement
        if reps >= 10:
            messages = base_messages + [
                f"Amazing! {reps} reps shows real dedication!",
                f"Incredible endurance! {reps} strong reps completed!",
                f"{random.choice(phrases['excellent'])}! You're crushing this workout!"
            ]
        elif reps >= 5:
            messages = base_messages + [
                f"Solid progress! {reps} reps with great form!",
                f"Halfway through and looking strong! {random.choice(phrases['keep_going'])}!",
                f"Building momentum! Your form is {random.choice(phrases['excellent'])}!"
            ]
        else:
            messages = base_messages + [
                f"Strong start! Focus on perfect form.",
                f"Great beginning! Establish that rhythm.",
                f"{random.choice(phrases['excellent'])} foundation! Build on this form."
            ]
    
    return random.choice(messages)

def get_rep_completion_message(reps, voice_config, ai_quota_exceeded=False):
    """Get AI-generated rep completion message with multilingual fallback."""
    
    if ai_quota_exceeded:
        return get_smart_rep_fallback(reps, voice_config)
    
    try:
        context = f"""
        You are an energetic fitness coach. The user just completed rep number {reps} of bicep curls.
        Voice: {voice_config['name']} ({voice_config['language']})
        Style: {voice_config.get('style', 'General')}
        
        Give a brief, enthusiastic congratulatory message (1 sentence, under 15 words).
        Make it sound natural and motivating for continued effort.
        Match the {voice_config.get('style', 'General')} style.
        """
        
        response = model.generate_content(context)
        if response.text:
            message = response.text.strip().replace('*', '').replace('#', '').replace('"', '')
            return message
        else:
            return get_smart_rep_fallback(reps, voice_config)
            
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return get_smart_rep_fallback(reps, voice_config)
        else:
            print(f"AI message error: {e}")
            return get_smart_rep_fallback(reps, voice_config)

def get_smart_rep_fallback(reps, voice_config):
    """Smart rep completion messages with language context."""
    import random
    
    # Language-specific congratulations
    lang_congrats = {
        'en-US': ['Excellent', 'Great', 'Outstanding', 'Perfect', 'Awesome'],
        'en-UK': ['Brilliant', 'Excellent', 'Superb', 'Well done', 'Splendid'],
        'hi-IN': ['Excellent', 'Bahut accha', 'Perfect', 'Shabash', 'Very good'],
        'zh-CN': ['Excellent', 'Great work', 'Perfect', 'Outstanding'],
        'fr-FR': ['Excellent', 'Parfait', 'Très bien', 'Magnifique'],
        'de-DE': ['Ausgezeichnet', 'Perfekt', 'Sehr gut', 'Excellent'],
        'es-ES': ['Excelente', 'Perfecto', 'Muy bien', 'Outstanding'],
        'it-IT': ['Eccellente', 'Perfetto', 'Molto bene', 'Fantastico']
    }
    
    congrats = lang_congrats.get(voice_config['lang_code'], lang_congrats['en-US'])
    
    if reps == 1:
        messages = [
            f"{random.choice(congrats)}! First rep with perfect form!",
            f"Great start! That's rep one in the books!",
            f"Perfect! Building strength from rep one!"
        ]
    elif reps % 10 == 0:
        messages = [
            f"{random.choice(congrats)}! {reps} reps milestone reached!",
            f"Incredible! That's {reps} strong reps completed!",
            f"Amazing dedication! {reps} reps shows real commitment!"
        ]
    elif reps % 5 == 0:
        messages = [
            f"Fantastic! {reps} reps with {random.choice(congrats).lower()} form!",
            f"Great milestone! {reps} reps of solid work!",
            f"{random.choice(congrats)} progress! {reps} controlled reps!"
        ]
    else:
        messages = [
            f"Great work! That's rep {reps} completed!",
            f"{random.choice(congrats)}! Rep {reps} with perfect form!",
            f"Outstanding! {reps} reps of quality training!",
            f"Perfect! Rep {reps} building real strength!"
        ]
    
    return random.choice(messages)

# ==============================================================================
# MAIN GYM CLASS
# ==============================================================================
class EnhancedVoiceGym:
    def __init__(self, voice_config):
        self.voice_config = voice_config
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
        self.start_time = time.time()
        self.ai_quota_exceeded = False  # Track AI quota status
        
        # Enhanced tracking
        self.workout_stats = {
            'peak_angles': [],
            'rep_times': [],
            'form_score': 100
        }
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open camera!")
            raise SystemExit()
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Enhanced VoiceGym initialized!")
        
    def process_frame(self, frame):
        """Process video frame for pose detection with AI feedback."""
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
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
                
                angle = calculate_angle(shoulder, elbow, wrist)
                current_time = time.time()
                workout_duration = (current_time - self.start_time) / 60
                
                # Rep counting with improved logic
                if angle > 160 and self.stage != "down":
                    self.stage = "down"
                elif angle < 50 and self.stage == "down" and current_time - self.last_rep > 2.0:
                    self.stage = "up"
                    self.reps += 1
                    self.last_rep = current_time
                    self.workout_stats['peak_angles'].append(angle)
                    self.workout_stats['rep_times'].append(current_time)
                    
                    # AI-generated rep completion message with fallback
                    if current_time - self.last_speech > 8:  # 8 second cooldown
                        rep_message = get_rep_completion_message(self.reps, self.voice_config, self.ai_quota_exceeded)
                        print(f"✅ Rep {self.reps}: {rep_message}")
                        speak_feedback(rep_message, self.voice_config)
                        self.last_speech = current_time
                    else:
                        print(f"✅ Rep {self.reps}! (Voice cooling down...)")
                
                # AI coaching feedback every 20 seconds with quota handling
                if (current_time - self.last_feedback > 20 and 
                    current_time - self.last_speech > 12):
                    
                    ai_tip = get_ai_coaching_feedback(angle, self.reps, self.stage, 
                                                    workout_duration, self.voice_config, self.ai_quota_exceeded)
                    if ai_tip:
                        coach_prefix = "🤖 AI Coach" if not self.ai_quota_exceeded else "💡 Smart Coach"
                        print(f"{coach_prefix} ({self.voice_config['name']}): {ai_tip}")
                        speak_feedback(ai_tip, self.voice_config)
                        self.last_feedback = current_time
                        self.last_speech = current_time
                
                # Enhanced UI overlay with multilingual support
                h, w = frame.shape[:2]
                
                # Background for stats
                cv2.rectangle(frame, (10, 10), (min(650, w-10), 170), (0,0,0), -1)
                
                # Title with language and voice info
                cv2.putText(frame, f'🏋️ Enhanced VoiceGym - AI Coach: {self.voice_config["name"]}', 
                           (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f'Language: {self.voice_config["language"]} | Style: {self.voice_config.get("style", "General")}', 
                           (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,255,255), 2)
                
                # Workout stats
                cv2.putText(frame, f'Reps: {self.reps} | Angle: {angle:.0f}° | Stage: {self.stage}', 
                           (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                cv2.putText(frame, f'Duration: {workout_duration:.1f}min | Voice ID: {self.voice_config["voice_id"]}', 
                           (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,255), 2)
                
                # Voice status with multilingual indicator
                time_since_speech = current_time - self.last_speech
                if time_since_speech < 10:
                    status_color = (100, 100, 255)  # Blue for cooling down
                    status_text = f'🔊 Voice cooldown: {10-time_since_speech:.1f}s'
                else:
                    status_color = (100, 255, 100)  # Green for ready
                    status_text = '🔊 AI Coach ready'
                    
                cv2.putText(frame, status_text, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2)
                
                # Form indicator
                if 40 <= angle <= 60 or 150 <= angle <= 170:
                    form_status = "✅ Perfect Form!"
                    form_color = (0, 255, 0)
                elif 30 <= angle <= 70 or 140 <= angle <= 180:
                    form_status = "⚠️  Good Form"
                    form_color = (0, 255, 255)
                else:
                    form_status = "⚡ Keep Focus"
                    form_color = (0, 165, 255)
                
                cv2.putText(frame, form_status, (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, form_color, 2)
                
                # Language-specific motivational indicator
                lang_flag = {
                    'en-US': '🇺🇸', 'en-UK': '🇬🇧', 'hi-IN': '🇮🇳', 'zh-CN': '🇨🇳',
                    'fr-FR': '🇫🇷', 'de-DE': '🇩🇪', 'es-ES': '🇪🇸', 'it-IT': '🇮🇹'
                }.get(self.voice_config['lang_code'], '🌍')
                
                cv2.putText(frame, f'{lang_flag} Multilingual Coach Active', (15, 155), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 2)
                
            except Exception as e:
                print(f"Pose processing error: {e}")
        
        return frame
    
    def run(self):
        """Main enhanced workout loop with multilingual support."""
        print("🎥 Starting Enhanced VoiceGym with Multilingual AI Coach...")
        print(f"🎤 Voice: {self.voice_config['name']} ({self.voice_config['language']})")
        print(f"🎭 Style: {self.voice_config.get('style', 'General')}")
        print(f"🆔 Voice ID: {self.voice_config['voice_id']}")
        print("🏋️ Position yourself for bicep curls and let the AI coach guide you!")
        print("📱 Press 'q' to quit or ESC to exit")
        print("=" * 70)
        
        frame_count = 0
        
        # Multilingual welcome messages
        welcome_messages = {
            'en-US': f"Welcome to your AI-powered workout! I'm {self.voice_config['name']}, your virtual fitness coach. Let's build strength together with perfect bicep curls!",
            'en-UK': f"Welcome to your personalised fitness session! I'm {self.voice_config['name']}, ready to guide your bicep curl workout. Let's achieve excellence together!",
            'hi-IN': f"Welcome to your AI fitness session! I'm {self.voice_config['name']}, your virtual coach. Let's do perfect bicep curls together!",
            'zh-CN': f"Welcome to your AI fitness training! I'm {self.voice_config['name']}, your virtual coach. Let's do excellent bicep curls together!",
            'fr-FR': f"Welcome to your AI fitness session! I'm {self.voice_config['name']}, your virtual coach. Let's do perfect bicep curls together!",
            'de-DE': f"Welcome to your AI fitness training! I'm {self.voice_config['name']}, your virtual coach. Let's do excellent bicep curls together!",
            'es-ES': f"Welcome to your AI fitness session! I'm {self.voice_config['name']}, your virtual coach. Let's do perfect bicep curls together!",
            'it-IT': f"Welcome to your AI fitness training! I'm {self.voice_config['name']}, your virtual coach. Let's do excellent bicep curls together!"
        }
        
        welcome_msg = welcome_messages.get(
            self.voice_config['lang_code'], 
            f"Welcome to your AI-powered workout! I'm {self.voice_config['name']}, your virtual fitness coach. Let's build strength together!"
        )
        
        print(f"🎯 {welcome_msg}")
        
        # Test voice with welcome message
        success = speak_feedback(welcome_msg, self.voice_config)
        if not success:
            print(f"⚠️  Voice synthesis had issues with {self.voice_config['name']}, but workout will continue with visual feedback.")
            print(f"🔧 Voice ID used: {self.voice_config['voice_id']}")
            print(f"🌍 Language Code: {self.voice_config['lang_code']}")
        else:
            print(f"✅ Voice test successful with {self.voice_config['name']}!")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("❌ Failed to grab frame from camera")
                break
            
            frame = cv2.flip(frame, 1)
            
            try:
                processed_frame = self.process_frame(frame)
                window_title = f'🏋️ Enhanced VoiceGym - {self.voice_config["language"]} Coach (Press Q to Quit)'
                cv2.imshow(window_title, processed_frame)
                
                frame_count += 1
                
                # Periodic stats with language info
                if frame_count % 300 == 0:  # Every ~10 seconds
                    elapsed = (time.time() - self.start_time) / 60
                    print(f"📊 Workout Stats: {self.reps} reps in {elapsed:.1f}min")
                    print(f"🎤 AI Coach: {self.voice_config['name']} ({self.voice_config['language']})")
                    print(f"🎭 Style: {self.voice_config.get('style', 'General')}")
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                cv2.imshow(f'🏋️ Enhanced VoiceGym - {self.voice_config["language"]} Coach (Press Q to Quit)', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        # Cleanup and multilingual final summary
        self.cap.release()
        cv2.destroyAllWindows()
        
        elapsed = (time.time() - self.start_time) / 60
        
        # AI-generated workout summary with multilingual context
        try:
            if not self.ai_quota_exceeded:
                summary_context = f"""
                Generate a brief, encouraging workout completion message for a {self.voice_config['language']} speaker. 
                The user completed {self.reps} bicep curls in {elapsed:.1f} minutes.
                Voice: {self.voice_config['name']} ({self.voice_config['language']})
                Style: {self.voice_config.get('style', 'General')}
                Keep it under 20 words and motivational.
                Match the {self.voice_config.get('style', 'General')} style.
                """
                
                response = model.generate_content(summary_context)
                if response.text:
                    final_message = response.text.strip().replace('*', '').replace('#', '').replace('"', '')
                else:
                    final_message = f"Fantastic workout! {self.reps} reps in {elapsed:.1f} minutes. You're building real strength with {self.voice_config['name']}!"
            else:
                final_message = f"Amazing job! {self.reps} bicep curls completed in {elapsed:.1f} minutes. Keep up the great work!"
                
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                self.ai_quota_exceeded = True
            
            # Language-specific completion messages
            completion_messages = {
                'en-US': f"Outstanding workout! {self.reps} reps in {elapsed:.1f} minutes. Excellent strength building session!",
                'en-UK': f"Brilliant workout! {self.reps} reps in {elapsed:.1f} minutes. Superb strength training session!",
                'hi-IN': f"Excellent workout! {self.reps} reps in {elapsed:.1f} minutes. Bahut accha strength building!",
                'zh-CN': f"Great workout! {self.reps} reps in {elapsed:.1f} minutes. Excellent strength training!",
                'fr-FR': f"Excellent workout! {self.reps} reps in {elapsed:.1f} minutes. Magnifique strength session!",
                'de-DE': f"Ausgezeichnet workout! {self.reps} reps in {elapsed:.1f} minutes. Perfect strength training!",
                'es-ES': f"Excelente workout! {self.reps} reps in {elapsed:.1f} minutes. Muy bien strength session!",
                'it-IT': f"Eccellente workout! {self.reps} reps in {elapsed:.1f} minutes. Fantastico strength training!"
            }
            
            final_message = completion_messages.get(
                self.voice_config['lang_code'],
                f"Outstanding workout! {self.reps} reps in {elapsed:.1f} minutes. Excellent session!"
            )
        
        print(f"🏁 {final_message}")
        print(f"🌍 Completed with {self.voice_config['name']} ({self.voice_config['language']})")
        speak_feedback(final_message, self.voice_config)
        
        # Keep program alive for final speech
        time.sleep(6)
        pygame.mixer.quit()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        print("🚀 Enhanced VoiceGym Coach with Multilingual AI Feedback")
        print("🌍 Supporting English, Hindi, Chinese, French, German, Spanish & Italian")
        print("=" * 70)
        
        # Get voice preferences with multilingual support
        voice_config = get_voice_settings()
        
        # Start enhanced multilingual workout
        gym = EnhancedVoiceGym(voice_config)
        gym.run()
        
    except KeyboardInterrupt:
        print("\n👋 Workout interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("🏁 Enhanced Multilingual VoiceGym Coach session ended!")