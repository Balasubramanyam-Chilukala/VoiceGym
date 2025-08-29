# VoiceGym: Multilingual AI Personal Trainer with Murf SDK Integration

A real-time, multilingual voice-enabled AI fitness coach powered by **Gemini AI**, **Murf SDK Translation**, and **Murf TTS API**. Features intelligent pose detection, dynamic AI coaching feedback, and authentic multilingual voice synthesis for bicep curl workouts.

This enhanced version transforms your webcam into an intelligent multilingual personal trainer using the **official Murf Python SDK** for seamless translation and voice generation. Experience the future of accessible, AI-driven home fitness with comprehensive voice personality options.

## 🌟 Key Features

### 🌍 Multilingual Support
- **6 Languages**: English (US), Hindi, Chinese (Mandarin), French, German, Spanish
- **25+ Voice Personalities**: Choose from male/female voices with authentic accents
- **Murf SDK Integration**: Official Murf Python SDK for reliable translation
- **Real-time Translation Pipeline**: Gemini AI → Murf Translation → Murf TTS → Audio

### 🤖 Advanced AI Coaching
- **Gemini AI-Powered Feedback**: Context-aware coaching with intelligent fallbacks
- **Dynamic Translation**: Real-time English-to-target-language conversion
- **Native Voice Synthesis**: Authentic multilingual voices via Murf TTS API
- **Smart Pipeline Management**: Handles API failures gracefully with manual translations

### 🏋️ Enhanced Workout Features
- **Intelligent Rep Detection**: Accurate counting with pose angle analysis
- **Form Analysis**: Real-time angle feedback (160° down, 50° up thresholds)
- **Workout Statistics**: Comprehensive tracking of reps, duration, and API usage
- **Pipeline Analytics**: Monitor Gemini calls, translations, and TTS generation

### 🎨 Rich Visual Interface
- **Pipeline Status Display**: Shows Gemini AI → Murf Translation → Murf TTS flow
- **Real-time Stats**: API usage tracking and pipeline performance metrics
- **Multilingual UI**: Language-specific voice information and coaching status
- **Error Resilience**: Visual feedback for API status and fallback activations

## 🛠️ Technology Stack

| Component | Technology / Library |
|-----------|---------------------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV, MediaPipe |
| AI Coaching Engine | Google Gemini API |
| Translation | **Murf Python SDK** |
| Voice Synthesis | **Murf TTS API** |
| Audio Management | Pygame (with threading) |
| Environment | python-dotenv |

## 🚀 Getting Started

### Prerequisites
- Python 3.8 - 3.10 (recommended)
- Webcam/camera device
- Internet connection (for AI and Murf APIs)

### ⚙️ Installation & Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/Balasubramanyam-Chilukala/VoiceGym.git
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
   - Choose from 6 languages and 25+ voice personalities
   - Interactive menu shows voice descriptions and styles
   - System uses Murf SDK for reliable voice selection

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
- **Hindi:**  Rahul, Amit, Kabir (Male), Shweta, Ayushi (Female) 
- **Chinese:** Zhang, Tao (Male), Jiao, Wei (Female)
- **French:** Maxime, Louis (Male), Adélie, Justine (Female)
- **German:** Matthias, Ralf (Male), Josephine, Lia (Female)
- **Spanish:** Javier (Male), Elvira, Carmen (Female)
- **Italian:** Lorenzo (Male), Greta (Female)

## 🔧 Advanced Features

### AI Coaching Intelligence
- **Context Awareness:** Analyzes current exercise phase, form quality, and progress
- **Murf SDK Translation:** Official SDK translates English to target language
- **Murf TTS Synthesis:** Authentic voice generation with selected personality
- **Audio Playback:** High-quality multilingual coaching delivery
- 
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
