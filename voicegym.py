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
import math
import random
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import stat

load_dotenv()
print("🏋️ Enhanced VoiceGym Coach - Fixed Murf SDK Integration Loading...")

# ==============================================================================
# SETUP
# ==============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

if not GEMINI_API_KEY or not MURF_API_KEY:
    print("❌ Please add your actual API keys to the .env file!")
    raise SystemExit()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
print("✅ API Keys configured!")

# ==============================================================================
# FIXED MURF SDK MANAGER
# ==============================================================================

class FixedMurfSDKManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.translation_cache = {}
        self.audio_cache = {}
        
        # Try to import Murf SDK
        try:
            from murf import Murf
            self.client = Murf(api_key=api_key)
            print("✅ Murf SDK initialized successfully!")
        except ImportError:
            print("📦 Installing Murf SDK...")
            os.system("pip install murf")
            try:
                from murf import Murf
                self.client = Murf(api_key=api_key)
                print("✅ Murf SDK installed and initialized!")
            except Exception as e:
                print(f"❌ Failed to install/import Murf SDK: {e}")
                print("⚠️ Using direct API calls as fallback...")
                self.client = None
                self._setup_direct_api()
        
        # Setup direct API as backup
        self._setup_direct_api()
        
        # Create a writable directory for audio files
        self.audio_dir = self._create_audio_directory()
        
        # Language mapping for Murf SDK
        self.lang_mapping = {
            'hi-IN': 'hi-IN',  # Hindi - India
            'zh-CN': 'zh-CN',  # Chinese - China
            'fr-FR': 'fr-FR',  # French - France
            'de-DE': 'de-DE',  # German - Germany
            'es-ES': 'es-ES',  # Spanish - Spain
            'es-MX': 'es-MX',  # Spanish - Mexico
            'it-IT': 'it-IT',  # Italian - Italy
            'ja-JP': 'ja-JP',  # Japanese - Japan
            'ko-KR': 'ko-KR',  # Korean - Korea
            'pt-BR': 'pt-BR',  # Portuguese - Brazil
            'nl-NL': 'nl-NL',  # Dutch - Netherlands
            'ta-IN': 'ta-IN',  # Tamil - India
            'bn-IN': 'bn-IN',  # Bengali - India
            'en-US': 'en-US',  # English - US
            'en-GB': 'en-UK',  # English - UK
            'en-IN': 'en-IN',  # English - India
            'en-AU': 'en-AU'   # English - Australia
        }
    
    def _create_audio_directory(self):
        """Create a writable directory for audio files"""
        try:
            # Try current directory first
            audio_dir = os.path.join(os.getcwd(), "murf_audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(audio_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            
            print(f"✅ Audio directory created: {audio_dir}")
            return audio_dir
            
        except Exception as e:
            print(f"⚠️ Cannot create audio directory in current folder: {e}")
            
            # Fallback to user's Documents folder
            try:
                import os.path
                documents_path = os.path.join(os.path.expanduser("~"), "Documents", "MurfVoiceGym")
                os.makedirs(documents_path, exist_ok=True)
                
                # Test write permissions
                test_file = os.path.join(documents_path, "test.txt")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                
                print(f"✅ Audio directory created in Documents: {documents_path}")
                return documents_path
                
            except Exception as e2:
                print(f"❌ Cannot create audio directory anywhere: {e2}")
                return tempfile.gettempdir()
    
    def _setup_direct_api(self):
        """Setup direct API calls"""
        self.base_url = "https://api.murf.ai/v1"
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def translate_with_murf_sdk(self, english_text, voice_config):
        """Fixed translation using Murf SDK with proper response parsing"""
        target_lang_code = voice_config['lang_code']
        target_lang = self.lang_mapping.get(target_lang_code, 'en-US')
        
        # Skip translation for English
        if target_lang.startswith('en-'):
            print(f"🔄 No translation needed for English: '{english_text[:50]}...'")
            return english_text
        
        cache_key = f"{english_text}_{target_lang}"
        if cache_key in self.translation_cache:
            print(f"🔄 Using cached translation")
            return self.translation_cache[cache_key]
        
        print(f"🌍 Translating to {target_lang} using Murf SDK: '{english_text[:40]}...'")
        
        # Method 1: Use Murf SDK with fixed response parsing
        if self.client:
            try:
                print("🔍 Using Murf SDK for translation...")
                
                response = self.client.text.translate(
                    target_language=target_lang,
                    texts=[english_text]
                )
                
                print(f"🔍 Murf SDK Response: {response}")
                
                # Fixed response parsing
                if response:
                    # Try different ways to access the translations
                    translations = None
                    
                    # Method 1: Direct access
                    if hasattr(response, 'translations') and response.translations:
                        translations = response.translations
                    # Method 2: Dictionary access
                    elif isinstance(response, dict) and 'translations' in response:
                        translations = response['translations']
                    # Method 3: Check if response itself is a list
                    elif isinstance(response, list):
                        translations = response
                    
                    if translations and len(translations) > 0:
                        # Extract translated text
                        first_translation = translations[0]
                        translated_text = None
                        
                        # Try different ways to get translated text
                        if hasattr(first_translation, 'translated_text'):
                            translated_text = first_translation.translated_text
                        elif isinstance(first_translation, dict):
                            translated_text = first_translation.get('translated_text')
                        
                        if translated_text and translated_text.strip():
                            self.translation_cache[cache_key] = translated_text
                            print(f"✅ Murf SDK Translation successful: '{translated_text[:40]}...'")
                            return translated_text
                
                print("❌ Murf SDK translation failed - invalid response format")
                
            except Exception as e:
                print(f"❌ Murf SDK translation error: {e}")
                print(f"🔍 Error type: {type(e)}")
        
        # Method 2: Direct API call as fallback
        try:
            print("🔍 Using direct Murf API as fallback...")
            return self._direct_api_translate(english_text, target_lang)
        except Exception as e:
            print(f"❌ Direct API translation error: {e}")
        
        # Method 3: Manual fallback translations
        return self._get_manual_translation(english_text, target_lang)
    
    def _direct_api_translate(self, text, target_lang):
        """Direct API translation call"""
        payload = {
            "target_language": target_lang,
            "texts": [text]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/text/translate",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            print(f"🔍 Direct API Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"🔍 Direct API Response: {result}")
                
                if 'translations' in result and result['translations']:
                    translated = result['translations'][0].get('translated_text')
                    if translated:
                        print(f"✅ Direct API translation: '{translated[:40]}...'")
                        return translated
                        
        except Exception as e:
            print(f"❌ Direct API failed: {e}")
        
        raise Exception("Direct API translation failed")
    
    def _get_manual_translation(self, text, target_lang):
        """Enhanced manual translations for fitness phrases"""
        print(f"🔧 Using manual translation fallback for {target_lang}")
        
        # Comprehensive Hindi fitness translations
        if target_lang == 'hi-IN':
            hindi_phrases = {
                # Complete phrase translations
                'boom! rep 3, crushing it!': 'बूम! रेप 3, शानदार!',
                'boom! rep 4, crushing it!': 'बूम! रेप 4, शानदार!',
                'crush those curls! welcome, shweta\'s got you!': 'बाइसेप कर्ल्स करें! स्वागत है, श्वेता आपके साथ है!',
                'strong curls! feel the burn, breathe deep!': 'मजबूत कर्ल्स! जलन महसूस करें, गहरी सांस लें!',
                
                # Word translations
                'boom': 'बूम',
                'crush': 'शानदार',
                'crushing': 'शानदार',
                'rep': 'रेप',
                'curls': 'कर्ल्स',
                'welcome': 'स्वागत है',
                'strong': 'मजबूत',
                'feel': 'महसूस करें',
                'burn': 'जलन',
                'breathe': 'सांस लें',
                'deep': 'गहरी',
                'excellent': 'उत्कृष्ट',
                'amazing': 'अद्भुत',
                'fantastic': 'शानदार',
                'great': 'बहुत बढ़िया',
                'perfect': 'परफेक्ट',
                'completed': 'पूरा हुआ',
                'done': 'हो गया',
                'it': 'इसे'
            }
            
            text_lower = text.lower()
            
            # Check for complete phrase first
            if text_lower in hindi_phrases:
                result = hindi_phrases[text_lower]
                print(f"✅ Complete phrase translation: '{result}'")
                return result
            
            # Word-by-word translation
            words = text_lower.split()
            translated_words = []
            
            for word in words:
                cleaned_word = word.strip('.,!?')
                if cleaned_word in hindi_phrases:
                    translated_words.append(hindi_phrases[cleaned_word])
                else:
                    translated_words.append(word)
            
            result = ' '.join(translated_words)
            print(f"✅ Manual translation: '{result}'")
            return result
        
        # For other languages, return original
        return text
    
    def generate_speech_with_murf_tts(self, text, voice_config):
        """Fixed speech generation with proper file handling"""
        voice_id = voice_config['voice_id']
        lang_code = voice_config['lang_code']
        
        cache_key = f"{text}_{voice_id}"
        if cache_key in self.audio_cache:
            print(f"🔄 Using cached TTS audio")
            return self.audio_cache[cache_key]
        
        try:
            print(f"🔊 Generating TTS with {voice_id}: '{text[:50]}...'")
            
            payload = {
                "text": text,
                "voiceId": voice_id,
                "format": "MP3",
                "model": "GEN2",
                "returnAsBase64": False,
                "language": lang_code,
                "speed": 1.0,
                "pitch": 0
            }
            
            response = requests.post(
                f"{self.base_url}/speech/generate",
                headers=self.headers,
                json=payload,
                timeout=20
            )
            
            print(f"🔍 TTS Status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                audio_url = response_data.get('audioFile')
                
                if audio_url:
                    print(f"🔗 Downloading TTS audio...")
                    audio_response = requests.get(audio_url, timeout=15)
                    
                    if audio_response.status_code == 200:
                        # Use the writable audio directory
                        audio_filename = os.path.join(
                            self.audio_dir, 
                            f"murf_tts_{voice_id}_{int(time.time())}_{random.randint(1000,9999)}.mp3"
                        )
                        
                        try:
                            with open(audio_filename, "wb") as f:
                                f.write(audio_response.content)
                            
                            # Set file permissions
                            os.chmod(audio_filename, stat.S_IREAD | stat.S_IWRITE)
                            
                            self.audio_cache[cache_key] = audio_filename
                            print(f"✅ Murf TTS generated: {os.path.basename(audio_filename)}")
                            return audio_filename
                            
                        except PermissionError as pe:
                            print(f"❌ Permission error saving audio: {pe}")
                            # Try with a different filename
                            alt_filename = os.path.join(
                                self.audio_dir,
                                f"audio_{int(time.time())}.mp3"
                            )
                            try:
                                with open(alt_filename, "wb") as f:
                                    f.write(audio_response.content)
                                return alt_filename
                            except:
                                print("❌ Cannot save audio file anywhere")
                                return None
            
            print(f"❌ TTS generation failed: {response.status_code}")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
        
        return None

# ==============================================================================
# GEMINI AI FEEDBACK GENERATOR
# ==============================================================================

class GeminiAIFeedbackGenerator:
    def __init__(self):
        self.model = model
        self.feedback_cache = {}
        self.ai_quota_exceeded = False
    
    def generate_rep_feedback(self, rep_count, exercise_name, voice_name):
        """Generate rep completion feedback using Gemini AI"""
        if self.ai_quota_exceeded:
            return self._get_fallback_rep_feedback(rep_count, exercise_name)
        
        try:
            prompt = f"""
            Generate a very short, enthusiastic fitness coaching message for completing rep {rep_count} of {exercise_name}.
            
            Requirements:
            - Maximum 6 words
            - Be energetic and motivational
            - Mention rep number
            - Use simple words for easy translation
            
            Examples:
            "Excellent! Rep {rep_count} completed!"
            "Amazing! Rep {rep_count} done!"
            "Perfect! Rep {rep_count} finished!"
            
            Generate one similar message:
            """
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                feedback = response.text.strip().replace('*', '').replace('"', '').replace('#', '')
                print(f"🤖 Gemini AI rep feedback: '{feedback}'")
                return feedback
                
        except Exception as e:
            if "quota" in str(e).lower():
                self.ai_quota_exceeded = True
                print("⚠️ Gemini AI quota exceeded")
            else:
                print(f"❌ Gemini AI error: {e}")
        
        return self._get_fallback_rep_feedback(rep_count, exercise_name)
    
    def generate_coaching_feedback(self, current_reps, exercise_name, voice_name):
        """Generate general coaching feedback using Gemini AI"""
        if self.ai_quota_exceeded:
            return self._get_fallback_coaching_feedback()
        
        try:
            prompt = f"""
            Generate a short motivational fitness coaching message during {exercise_name} workout.
            
            Requirements:
            - Maximum 8 words
            - Focus on form, breathing, or encouragement
            - Be energetic and supportive
            - Use simple words for easy translation
            
            Examples:
            "Great form! Focus on breathing!"
            "Excellent work! Keep steady rhythm!"
            "Perfect! You're getting stronger!"
            
            Generate one similar message:
            """
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                feedback = response.text.strip().replace('*', '').replace('"', '').replace('#', '')
                print(f"🤖 Gemini AI coaching: '{feedback}'")
                return feedback
                
        except Exception as e:
            if "quota" in str(e).lower():
                self.ai_quota_exceeded = True
                print("⚠️ Gemini AI quota exceeded")
            else:
                print(f"❌ Gemini AI error: {e}")
        
        return self._get_fallback_coaching_feedback()
    
    def generate_welcome_message(self, voice_name, exercise_name):
        """Generate welcome message using Gemini AI"""
        if self.ai_quota_exceeded:
            return f"Welcome! Ready for {exercise_name}!"
        
        try:
            prompt = f"""
            Generate a short, energetic welcome message for starting a {exercise_name} workout with AI coach {voice_name}.
            
            Requirements:
            - Maximum 8 words
            - Be welcoming and motivational
            - Use simple words for easy translation
            
            Generate a welcome message:
            """
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                welcome = response.text.strip().replace('*', '').replace('"', '').replace('#', '')
                print(f"🤖 Gemini AI welcome: '{welcome}'")
                return welcome
                
        except Exception as e:
            if "quota" in str(e).lower():
                self.ai_quota_exceeded = True
        
        return f"Welcome! Ready for {exercise_name}!"
    
    def _get_fallback_rep_feedback(self, rep_count, exercise_name):
        """Fallback rep feedback messages"""
        messages = [
            f"Excellent! Rep {rep_count} completed!",
            f"Amazing! Rep {rep_count} done!",
            f"Perfect! Rep {rep_count} finished!",
            f"Great! Rep {rep_count} complete!",
            f"Fantastic! Rep {rep_count} done!"
        ]
        return random.choice(messages)
    
    def _get_fallback_coaching_feedback(self):
        """Fallback coaching feedback messages"""
        messages = [
            "Great form! Keep breathing!",
            "Excellent work! Stay strong!",
            "Perfect! You're getting stronger!",
            "Amazing! Focus on form!",
            "Outstanding! Keep it up!",
            "Fantastic! Breathe deeply!",
            "Superb! Maintain rhythm!"
        ]
        return random.choice(messages)

# ==============================================================================
# HARDCODED VOICES DATABASE
# ==============================================================================

def get_murf_voices_database():
    """Complete Murf voices database"""
    return {
        "Hindi (India)": {
            "hi-IN": {
                "Male": [
                    {"id": "hi-IN-rahul", "name": "Rahul"},
                    {"id": "hi-IN-amit", "name": "Amit"},
                    {"id": "hi-IN-shaan", "name": "Shaan"},
                    {"id": "hi-IN-kabir", "name": "Kabir"}
                ],
                "Female": [
                    {"id": "hi-IN-shweta", "name": "Shweta"},
                    {"id": "hi-IN-ayushi", "name": "Ayushi"}
                ]
            }
        },
        "Chinese (Mandarin)": {
            "zh-CN": {
                "Male": [
                    {"id": "zh-CN-zhang", "name": "Zhang"},
                    {"id": "zh-CN-tao", "name": "Tao"}
                ],
                "Female": [
                    {"id": "zh-CN-jiao", "name": "Jiao"},
                    {"id": "zh-CN-wei", "name": "Wei"}
                ]
            }
        },
        "French (France)": {
            "fr-FR": {
                "Male": [
                    {"id": "fr-FR-maxime", "name": "Maxime"},
                    {"id": "fr-FR-louis", "name": "Louis"}
                ],
                "Female": [
                    {"id": "fr-FR-adélie", "name": "Adélie"},
                    {"id": "fr-FR-justine", "name": "Justine"}
                ]
            }
        },
        "English (US)": {
            "en-US": {
                "Male": [
                    {"id": "en-US-ken", "name": "Ken"},
                    {"id": "en-US-ryan", "name": "Ryan"}
                ],
                "Female": [
                    {"id": "en-US-natalie", "name": "Natalie"},
                    {"id": "en-US-samantha", "name": "Samantha"}
                ]
            }
        }
    }

# ==============================================================================
# FIXED PIPELINE AUDIO SYSTEM
# ==============================================================================

class FixedPipelineAudioSystem:
    def __init__(self, voice_config, gemini_ai, murf_manager):
        self.voice_config = voice_config
        self.gemini_ai = gemini_ai
        self.murf = murf_manager
        self.audio_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.playing = False
        self._shutdown = False
    
    def speak_with_fixed_pipeline(self, feedback_type, priority=False, **kwargs):
        """Fixed complete pipeline: Gemini AI → Murf SDK Translation → Murf TTS → Audio"""
        if self._shutdown:
            return
            
        if priority:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break
        
        self.audio_queue.put((feedback_type, kwargs, priority))
        self.executor.submit(self._process_fixed_pipeline)
        return True
    
    def _process_fixed_pipeline(self):
        """Process the fixed pipeline"""
        if self.playing or self._shutdown:
            return
            
        try:
            feedback_type, kwargs, priority = self.audio_queue.get_nowait()
            
            print(f"\n🎯 FIXED MURF SDK PIPELINE: {feedback_type}")
            print("=" * 70)
            
            # STEP 1: Generate English feedback using Gemini AI
            print("🤖 STEP 1: Gemini AI generating English feedback...")
            english_text = self._generate_english_feedback(feedback_type, **kwargs)
            print(f"✅ Gemini AI Output: '{english_text}'")
            
            # STEP 2: Translate using fixed Murf SDK
            print("🌍 STEP 2: Fixed Murf SDK Translation...")
            translated_text = self.murf.translate_with_murf_sdk(english_text, self.voice_config)
            print(f"✅ Fixed Translation: '{translated_text}'")
            
            # STEP 3: Generate speech using Murf TTS API
            print("🔊 STEP 3: Murf TTS generation...")
            audio_file = self.murf.generate_speech_with_murf_tts(translated_text, self.voice_config)
            
            # STEP 4: Play audio
            if audio_file:
                print("🎵 STEP 4: Playing native language audio...")
                self._play_audio_file(audio_file)
                print(f"✅ FIXED PIPELINE COMPLETE: {self.voice_config['name']} spoke in {self.voice_config['language']}!")
            else:
                print("❌ PIPELINE FAILED: No audio generated")
            
            print("=" * 70)
            
        except queue.Empty:
            pass
        except Exception as e:
            if not self._shutdown:
                print(f"❌ Fixed pipeline error: {e}")
    
    def _generate_english_feedback(self, feedback_type, **kwargs):
        """Generate English feedback using Gemini AI"""
        if feedback_type == 'rep_completed':
            return self.gemini_ai.generate_rep_feedback(
                kwargs.get('rep', 1), 
                kwargs.get('exercise', 'Bicep Curls'),
                self.voice_config['name']
            )
        elif feedback_type == 'coaching':
            return self.gemini_ai.generate_coaching_feedback(
                kwargs.get('reps', 0),
                kwargs.get('exercise', 'Bicep Curls'),
                self.voice_config['name']
            )
        elif feedback_type == 'welcome':
            return self.gemini_ai.generate_welcome_message(
                self.voice_config['name'],
                kwargs.get('exercise', 'Bicep Curls')
            )
        elif feedback_type == 'paused':
            return "Workout paused. Resume when ready!"
        elif feedback_type == 'resume':
            return "Welcome back! Let's continue!"
        else:
            return "Great work! Keep it up!"
    
    def _play_audio_file(self, audio_filename):
        """Play audio file safely with enhanced error handling"""
        try:
            self.playing = True
            
            # Reinitialize pygame mixer if needed
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Check if file exists and is readable
            if not os.path.exists(audio_filename):
                print(f"❌ Audio file does not exist: {audio_filename}")
                return
            
            if not os.access(audio_filename, os.R_OK):
                print(f"❌ Audio file not readable: {audio_filename}")
                return
            
            pygame.mixer.music.load(audio_filename)
            pygame.mixer.music.play()
            
            timeout = time.time() + 30
            while pygame.mixer.music.get_busy() and time.time() < timeout and not self._shutdown:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
        finally:
            self.playing = False
    
    def shutdown(self):
        """Shutdown safely"""
        self._shutdown = True
        self.playing = False

# ==============================================================================
# VOICE SELECTION
# ==============================================================================

def select_voice():
    """Select voice from database"""
    print("\n🎤 SELECT YOUR MURF AI FITNESS COACH")
    print("=" * 60)
    
    voices_db = get_murf_voices_database()
    voice_options = []
    counter = 1
    
    for language, lang_codes in voices_db.items():
        print(f"\n🌍 {language}:")
        for lang_code, genders in lang_codes.items():
            for gender, voices in genders.items():
                print(f"  👤 {gender}:")
                for voice in voices:
                    print(f"    {counter}. {voice['name']}")
                    voice_options.append({
                        'language': language,
                        'lang_code': lang_code,
                        'voice_id': voice['id'],
                        'name': voice['name'],
                        'gender': gender
                    })
                    counter += 1
    
    print(f"\n💫 Choose your coach (1-{len(voice_options)}): ", end="")
    
    while True:
        try:
            choice = int(input())
            if 1 <= choice <= len(voice_options):
                selected = voice_options[choice - 1]
                print(f"\n✅ Selected: {selected['name']} ({selected['language']})")
                print(f"   🎭 Voice ID: {selected['voice_id']}")
                print(f"   🌍 Language Code: {selected['lang_code']}")
                return selected
            else:
                print(f"❌ Enter 1-{len(voice_options)}: ", end="")
        except ValueError:
            print("❌ Enter a number: ", end="")

# ==============================================================================
# MAIN GYM CLASS
# ==============================================================================

class FixedMurfSDKVoiceGym:
    def __init__(self, voice_config, audio_system):
        self.voice_config = voice_config
        self.audio_system = audio_system
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Core tracking
        self.reps = 0
        self.stage = "ready"
        self.last_feedback = 0
        self.last_rep = 0
        self.last_speech = 0
        self.start_time = time.time()
        
        # Exercise
        self.exercise = {
            'name': 'Bicep Curls',
            'landmarks': [11, 13, 15],
            'down_threshold': 160,
            'up_threshold': 50
        }
        
        # Stats tracking
        self.stats = {
            'gemini_calls': 0,
            'murf_translations': 0,
            'murf_tts_calls': 0,
            'total_pipelines': 0
        }
        
        self.is_paused = False
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise Exception("Cannot open camera!")
        
        print("✅ Fixed Murf SDK VoiceGym initialized!")
    
    def calculate_angle(self, landmarks):
        """Calculate angle for bicep curls"""
        try:
            indices = self.exercise['landmarks']
            a = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
            b = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
            c = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180 else angle
        except:
            return 90
    
    def speak_with_fixed_pipeline(self, feedback_type, priority=False, **kwargs):
        """Use fixed pipeline and track stats"""
        self.audio_system.speak_with_fixed_pipeline(feedback_type, priority, **kwargs)
        self.stats['gemini_calls'] += 1
        if self.voice_config['lang_code'] not in ['en-US', 'en-GB']:
            self.stats['murf_translations'] += 1
        self.stats['murf_tts_calls'] += 1
        self.stats['total_pipelines'] += 1
    
    def process_frame(self, frame):
        """Process frame with pose detection"""
        current_time = time.time()
        
        if self.is_paused:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, 'PAUSED - Press P to resume', (w//2 - 200, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame
        
        # Pose detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        angle = 90
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angle = self.calculate_angle(landmarks)
            
            # Rep counting
            if angle > self.exercise['down_threshold'] and self.stage != "down":
                self.stage = "down"
            elif (angle < self.exercise['up_threshold'] and 
                  self.stage == "down" and 
                  current_time - self.last_rep > 2.0):
                self.stage = "up"
                self.reps += 1
                self.last_rep = current_time
                
                # Rep completion feedback
                if current_time - self.last_speech > 5:
                    self.speak_with_fixed_pipeline('rep_completed', 
                                                 rep=self.reps, 
                                                 exercise=self.exercise['name'])
                    self.last_speech = current_time
                    print(f"✅ Rep {self.reps} completed!")
            
            # Coaching feedback
            if (current_time - self.last_feedback > 25 and 
                current_time - self.last_speech > 10):
                
                self.speak_with_fixed_pipeline('coaching',
                                             reps=self.reps,
                                             exercise=self.exercise['name'])
                self.last_feedback = current_time
                self.last_speech = current_time
            
            # Draw pose
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255,0,255), thickness=2)
            )
        
        return self.draw_overlay(frame, angle)
    
    def draw_overlay(self, frame, angle):
        """Draw overlay with stats"""
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (10, 10), (min(800, w-10), 140), (0,0,0), -1)
        
        cv2.putText(frame, f'🏋️ Fixed Murf SDK VoiceGym - Working Translation', 
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f'🎤 {self.voice_config["name"]} | 🌍 {self.voice_config["language"]} | 🆔 {self.voice_config["voice_id"]}', 
                   (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,255,255), 2)
        cv2.putText(frame, f'Exercise: {self.exercise["name"]} | Reps: {self.reps} | Angle: {angle:.0f}° | Stage: {self.stage}', 
                   (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        cv2.putText(frame, f'🤖 Gemini: {self.stats["gemini_calls"]} | 🌍 Murf Translation: {self.stats["murf_translations"]} | 🔊 Murf TTS: {self.stats["murf_tts_calls"]} | 🎯 Total: {self.stats["total_pipelines"]}', 
                   (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 2)
        cv2.putText(frame, 'Pipeline: Gemini AI → Fixed Murf SDK → Murf TTS → Native Audio | P=Pause, Q=Quit', 
                   (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
        
        return frame
    
    def run_fixed_workout(self):
        """Main workout loop with fixed Murf SDK"""
        print("🎥 Starting Fixed Murf SDK VoiceGym...")
        print(f"🎤 Voice: {self.voice_config['name']} ({self.voice_config['language']})")
        print(f"🆔 Voice ID: {self.voice_config['voice_id']}")
        print("🎯 Pipeline: Gemini AI → Fixed Murf SDK → Murf TTS → Native Audio")
        print("⌨️ Controls: P=Pause, Q=Quit")
        print("=" * 80)
        
        # Welcome message
        print("🚀 Starting fixed welcome pipeline...")
        self.speak_with_fixed_pipeline('welcome', priority=True, exercise=self.exercise['name'])
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('🏋️ Fixed Murf SDK VoiceGym - Perfect Translation', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.speak_with_fixed_pipeline('paused', priority=True)
                else:
                    self.speak_with_fixed_pipeline('resume', priority=True)
            
            frame_count += 1
            if frame_count % 300 == 0:
                print(f"📊 Frame: {frame_count} | Reps: {self.reps} | Fixed Pipelines: {self.stats['total_pipelines']}")
        
        # Cleanup
        self.audio_system.shutdown()
        self.cap.release()
        cv2.destroyAllWindows()
        
        elapsed = (time.time() - self.start_time) / 60
        
        print(f"\n🏁 Fixed Workout Complete!")
        print(f"   Reps: {self.reps}")
        print(f"   Duration: {elapsed:.1f} minutes")
        print(f"   Fixed Pipeline runs: {self.stats['total_pipelines']}")
        print(f"   Murf translations: {self.stats['murf_translations']}")
        print(f"   TTS calls: {self.stats['murf_tts_calls']}")
        
        time.sleep(3)
        try:
            pygame.mixer.quit()
        except:
            pass

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        print("🚀 Fixed Murf SDK VoiceGym - Perfect Integration")
        print("🔥 Gemini AI → Fixed Murf SDK → Murf TTS")
        print("🌍 Perfect multilingual AI fitness coaching")
        print("=" * 80)
        
        # Select voice
        voice_config = select_voice()
        
        if not voice_config:
            print("❌ Voice configuration failed")
            exit(1)
        
        # Initialize systems
        print("\n🔧 Initializing fixed systems...")
        gemini_ai = GeminiAIFeedbackGenerator()
        murf_manager = FixedMurfSDKManager(MURF_API_KEY)
        
        print("✅ All systems initialized!")
        
        # Create fixed audio system
        audio_system = FixedPipelineAudioSystem(voice_config, gemini_ai, murf_manager)
        
        # Create gym
        gym = FixedMurfSDKVoiceGym(voice_config, audio_system)
        
        # Start fixed workout
        gym.run_fixed_workout()
        
    except KeyboardInterrupt:
        print("\n👋 Workout interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
            pygame.mixer.quit()
        except:
            pass
        print("🏁 Fixed Murf SDK VoiceGym session ended!")
        print("🎯 Perfect: Gemini AI + Murf SDK + Murf TTS")
