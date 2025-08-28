# VoiceGym: Multilingual AI Personal Trainer

> A real-time, multilingual voice-enabled AI fitness coach that runs entirely on your local machine. Features 8 languages, 20+ voice personalities, intelligent pose detection, and dynamic AI coaching feedback for bicep curl workouts.

This enhanced version transforms your webcam into an intelligent multilingual personal trainer, offering natural voice coaching in English (US/UK), Hindi, Chinese, French, German, Spanish, and Italian. It demonstrates the future of accessible, AI-driven home fitness with comprehensive voice personality options.

## 🌟 Key Features

### 🌍 Multilingual Support
- **8 Languages:** English (US/UK), Hindi, Chinese (Mandarin), French, German, Spanish, Italian
- **10+ Voice Personalities:** Choose from male/female voices with different styles (Conversational, Promo, General)
- **Dynamic Voice Menu:** Interactive selection system with language-specific voice descriptions
- **Real-time API Integration:** Fetches live voice options from Murf AI API

### 🤖 Advanced AI Coaching
- **Context-Aware Feedback:** AI analyzes your current angle, rep count, workout duration, and stage
- **Language-Specific Coaching Style:** Adapts coaching tone and approach based on selected language/voice
- **Smart Fallback System:** Continues with intelligent coaching even if AI quota is exceeded
- **Dynamic Rep Celebrations:** AI-generated congratulations for milestone achievements

### 🏋️ Enhanced Workout Features
- **Improved Rep Detection:** More accurate counting with 2-second cooldown to prevent double counting
- **Form Analysis:** Real-time angle feedback with perfect form indicators
- **Workout Statistics:** Tracks peak angles, rep timing, and form scores
- **Voice Cooldown Management:** Prevents audio overlap with intelligent spacing

### 🎨 Rich Visual Interface
- **Multilingual UI:** Display shows selected language, voice personality, and coaching status
- **Real-time Stats:** Comprehensive overlay with reps, angles, duration, and voice information
- **Form Quality Indicators:** Visual feedback for perfect form, good form, and focus zones
- **Language Flags:** Visual indicators showing active coaching language

## 🛠️ Technology Stack

| Component | Technology / Library |
|-----------|---------------------|
| **Language** | Python 3.8+ |
| **Computer Vision** | OpenCV, MediaPipe |
| **AI Coaching Engine** | Google Gemini API (with smart fallbacks) |
| **Voice Synthesis** | Murf AI API (comprehensive voice library) |
| **Audio Management** | Pygame (with async threading) |
| **Environment** | python-dotenv for secure API key management |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 - 3.10 (recommended)
- Webcam/camera device
- Internet connection (for AI and voice APIs)
- Microphone permissions (optional, for voice commands)

### ⚙️ Installation & Setup

1. **Clone or Download the Repository:**
   ```bash
   git clone <your-repo-url>
   cd VoiceGym
   ```

2. **Set Up Virtual Environment:**
   ```bash
   # Using Conda
   conda create --name voicegym python=3.9 -y
   conda activate voicegym
   
   # Or using venv
   python -m venv voicegym-env
   source voicegym-env/bin/activate  # On Windows: enhanced-voicegym-env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy requests pygame python-dotenv google-generativeai
   ```

### 🔑 API Configuration

1. **Create Environment File:**
   Create a `.env` file in your project directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   MURF_API_KEY=your_murf_api_key_here
   ```

2. **Get API Keys:**
   - **Gemini API Key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Murf API Key:** Visit [Murf AI API](https://murf.ai/api)

3. **API Key Security:**
   - Never commit `.env` files to version control
   - The script validates API keys on startup
   - Both keys are required for full functionality

## ▶️ How to Run

1. **Activate Environment:**
   ```bash
   conda activate voicegym  # or source your venv
   ```

2. **Launch Application:**
   ```bash
   python voicegym.py
   ```

3. **Select Your AI Coach:**
   - Choose from 8 languages and 10+ voice personalities
   - Interactive menu shows voice descriptions and styles
   - System fetches real-time voice options from Murf API

4. **Start Your Workout:**
   - Position yourself so the camera can see your upper body
   - Begin performing bicep curls
   - Your AI coach will provide real-time feedback and encouragement

5. **Exit:**
   - Press 'q' or 'ESC' while the camera window is active
   - AI coach will provide a personalized workout summary

## 🎤 Voice Personalities

### English Options
- **Ken (US)** - Conversational American male
- **Carter (US)** - Multilingual conversational (supports French)
- **Natalie (US)** - Promotional female style
- **Ruby (UK)** - Conversational British (supports German)

### International Options
- **Hindi:** Amit (Male), Ayushi (Female)
- **Chinese:** Tao (Male), Jiao (Female)
- **French:** Adélie (Female)
- **German:** Matthias (Male)
- **Spanish:** Javier (Male), Elvira (Female)
- **Italian:** Lorenzo (Male), Greta (Female)

## 🔧 Advanced Features

### AI Coaching Intelligence
- **Context Awareness:** Analyzes current exercise phase, form quality, and progress
- **Adaptive Feedback:** Adjusts coaching style based on selected voice personality
- **Progress Tracking:** Monitors workout duration, rep consistency, and form improvements
- **Motivational Milestones:** Special encouragement for rep achievements (5, 10, 15+ reps)

### Smart Fallback System
- **Quota Management:** Automatically switches to intelligent fallback coaching if AI limits are reached
- **Language-Aware Fallbacks:** Maintains language-appropriate motivational phrases
- **Seamless Transition:** Users continue receiving quality coaching without interruption

### Technical Optimizations
- **Async Audio Processing:** Non-blocking voice synthesis and playback
- **Frame Rate Optimization:** Maintains smooth video processing during voice generation
- **Memory Management:** Efficient temporary file handling for audio clips
- **Error Resilience:** Graceful handling of API timeouts and network issues

## 🏆 Workout Analytics

The system tracks comprehensive workout metrics:
- **Rep Count & Timing:** Precise counting with time stamps
- **Form Quality Scores:** Real-time analysis of exercise form
- **Peak Angle Tracking:** Records optimal contraction points
- **Workout Duration:** Total session time with per-rep timing
- **Voice Interaction Stats:** Coaching frequency and response times

## 🔍 Troubleshooting

### Common Issues
- **Camera Not Found:** Check camera permissions and ensure no other applications are using the camera
- **API Key Errors:** Verify keys in `.env` file and check API quotas
- **Voice Synthesis Fails:** Check internet connection and Murf API status
- **Poor Pose Detection:** Ensure good lighting and clear view of upper body

### Performance Tips
- Use good lighting for better pose detection
- Position camera at chest level for optimal angle calculation
- Allow 2-3 seconds between reps for accurate counting
- Keep upper body centered in camera frame

## 🌐 Language Support Status

| Language | Voice Count | TTS Quality | AI Coaching | Status |
|----------|-------------|-------------|-------------|--------|
| English (US) | 3 | Excellent | Full Support | ✅ Complete |
| English (UK) | 1 | Excellent | Full Support | ✅ Complete |
| Hindi | 2 | Very Good | Full Support | ✅ Complete |
| Chinese | 2 | Very Good | Full Support | ✅ Complete |
| French | 1 | Very Good | Full Support | ✅ Complete |
| German | 1 | Very Good | Full Support | ✅ Complete |
| Spanish | 2 | Very Good | Full Support | ✅ Complete |
| Italian | 2 | Very Good | Full Support | ✅ Complete |


## 📄 License

This project is for educational and personal use. Please respect the terms of service for Gemini API and Murf API when using their services.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional language support
- New exercise recognition
- Enhanced AI coaching logic
- Performance optimizations
- UI/UX improvements

---

**Built with ❤️ for the future of accessible, multilingual fitness coaching**
