# 🏋️ VoiceGym: Your AI Personal Trainer

> A revolutionary real-time, voice-enabled AI fitness coach with a modern Streamlit web interface. Uses your webcam, computer vision, and generative AI to analyze your exercise form, count your reps, and provide live, personalized verbal feedback.

This project transforms your webcam into an intelligent personal trainer with both a traditional command-line interface and a modern web-based UI. It demonstrates the future of accessible, AI-driven home fitness with customizable coaching styles and multi-language support.

![VoiceGym Streamlit Interface](https://github.com/user-attachments/assets/8d5b3dfb-1ad1-4005-ae08-0121332dadd6)

## 🌟 Key Features

### 🆕 New Streamlit Web Interface
-   **Modern Web UI**: Beautiful, responsive Streamlit interface with real-time video feed
-   **Configuration Panel**: Easy setup of API keys, voice preferences, and coaching styles
-   **Live Statistics**: Real-time workout stats, progress charts, and session history
-   **Multi-language Support**: Choose from multiple languages for voice feedback
-   **Voice Selection**: Pick from various voices with different accents and genders

### 🤖 Advanced AI Integration
-   **Dynamic AI Feedback**: Powered by Google Gemini AI for contextual, personalized coaching
-   **Coaching Styles**: Choose from Gentle/Encouraging, Motivational, or High-intensity coaching
-   **Smart Context Awareness**: AI understands your progress, form, and workout phase
-   **Adaptive Responses**: Feedback adapts to your performance and improvement over time

### 🎯 Core Exercise Tracking
-   **Real-Time Pose Estimation**: Utilizes Google's MediaPipe library for precise body tracking
-   **Live Voice Coaching**: Natural-sounding speech powered by Murf AI API
-   **Automatic Rep Counting**: Biomechanical angle-based counting for bicep curls
-   **Form Analysis**: Instant feedback on exercise technique and range of motion
-   **Session Analytics**: Track progress, reps per minute, and workout duration

### 🔧 Technical Excellence
-   **Asynchronous Architecture**: Smooth video feed with non-blocking audio feedback
-   **Modular Design**: Clean separation between UI, core logic, and AI services
-   **Error Handling**: Graceful fallbacks when APIs are unavailable
-   **Cross-Platform**: Works on Windows, macOS, and Linux

## 🛠️ Technology Stack

| Component                | Technology / Library                                       |
| :----------------------- | :--------------------------------------------------------- |
| **Web Interface**        | Streamlit                                                  |
| **Language**             | Python 3.8+                                               |
| **Computer Vision**      | OpenCV, MediaPipe                                          |
| **AI Feedback Engine**   | Google Gemini API                                          |
| **Voice Synthesis (TTS)**| Murf AI API                                                |
| **Audio Playback**       | Pygame                                                     |
| **Data Visualization**   | Plotly, Pandas                                             |
| **Async Processing**     | asyncio, aiohttp                                           |
| **Environment**          | pip, virtual environments                                  |

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

- **Python 3.8 or higher**
- **Webcam** (built-in or external)
- **Internet connection** for AI services
- **API Keys**:
  - [Google Gemini API Key](https://makersuite.google.com/app/apikey) (Free tier available)
  - [Murf AI API Key](https://murf.ai/api) (Free trial available)

### ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Balasubramanyam-Chilukala/VoiceGym.git
   cd VoiceGym
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API keys**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your API keys
   nano .env  # or use your preferred editor
   ```

   Add your keys to the `.env` file:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   MURF_API_KEY=your_murf_api_key_here
   ```

### 🔑 Getting API Keys

#### Gemini AI API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file

#### Murf AI API Key
1. Visit [Murf AI API](https://murf.ai/api)
2. Sign up for an account
3. Navigate to API settings
4. Generate an API key
5. Copy the key to your `.env` file

## ▶️ How to Run

### Option 1: Modern Streamlit Web Interface (Recommended)

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Configure your settings** in the sidebar:
   - Enter your API keys
   - Select your preferred voice
   - Choose coaching style
   - Adjust workout settings

4. **Start your workout** by clicking "🚀 Start Workout"

5. **Position yourself** so the camera can see your upper body and begin bicep curls

### Option 2: Enhanced Command Line Interface

1. **Run the enhanced script**
   ```bash
   python voicegym_enhanced.py
   ```

2. **Select your preferences** when prompted:
   - Choose coaching style (Gentle/Motivational/High-intensity)
   - Select voice preference
   - Confirm settings

3. **Start exercising** when the camera window opens

### Option 3: Original Interface (Legacy)

1. **Run the original script** (backwards compatible)
   ```bash
   python voicegym.py
   ```

2. **Begin exercising** immediately with default settings

To stop any version, press **'q'** or **ESC** key when the camera window is active.

## 🎯 Coaching Styles

### 💙 Gentle/Encouraging
- **Tone**: Supportive, patient, nurturing
- **Best for**: Beginners, rehabilitation, stress relief
- **Example**: *"Wonderful! That's rep 5. You're doing beautifully! Take your time and focus on your breathing."*

### 🔥 Motivational (Default)
- **Tone**: Energetic, uplifting, achievement-focused
- **Best for**: General fitness, goal achievement
- **Example**: *"Fantastic! That's rep 5! You're crushing this workout and building serious strength!"*

### ⚡ High-Intensity
- **Tone**: Direct, challenging, high-energy
- **Best for**: Advanced athletes, competition prep
- **Example**: *"BOOM! Rep 5 CRUSHED! You're an absolute machine! KEEP PUSHING!"*

## 📊 Features Overview

### Web Interface Features
- ✅ Real-time camera feed with pose overlay
- ✅ Live workout statistics and progress tracking
- ✅ Session history with exportable data
- ✅ Interactive charts and analytics
- ✅ Configurable voice and language settings
- ✅ Multiple coaching style options
- ✅ API key management interface

### AI Integration Features
- ✅ Context-aware feedback based on exercise phase
- ✅ Personalized coaching adapted to your form
- ✅ Progressive difficulty and encouragement
- ✅ Multilingual voice synthesis
- ✅ Smart timing for feedback delivery
- ✅ Fallback modes when APIs are unavailable

### Exercise Tracking Features
- ✅ Precise bicep curl detection and counting
- ✅ Range of motion analysis
- ✅ Form feedback and corrections
- ✅ Session duration and rate tracking
- ✅ Rep history and progress visualization

## 🔧 Configuration Options

### Voice Settings
- **Available Voices**: Multiple English voices (US/UK/AU), plus Spanish, French, German, Italian
- **Languages**: Support for 9+ languages with localized coaching
- **Quality**: High-quality neural voice synthesis

### Workout Settings
- **Feedback Frequency**: Adjustable coaching intervals (10-60 seconds)
- **Rep Announcements**: Toggle individual rep counting
- **Coaching Intensity**: Three distinct coaching personalities
- **Exercise Focus**: Currently optimized for bicep curls (extensible)

### Technical Settings
- **Camera Resolution**: 640x480 (configurable)
- **Detection Confidence**: Adjustable pose detection sensitivity
- **Audio Quality**: High-quality MP3 synthesis and playback

## 🧪 Testing & Development

### Running Tests
```bash
# Test core functionality
python -c "from voicegym_core import VoiceGymCore; print('Core module working!')"

# Test Streamlit app
streamlit run app.py --server.headless true

# Test enhanced CLI
python voicegym_enhanced.py
```

### Development Mode
- Set `DEBUG=True` in your `.env` file for verbose logging
- Use fallback modes to develop without API keys
- Camera simulation mode available for headless testing

## 🤝 Contributing

We welcome contributions! Here are ways you can help:

- 🐛 **Bug Reports**: Found an issue? Report it!
- 💡 **Feature Requests**: Have an idea? Share it!
- 🔧 **Code Contributions**: Submit pull requests
- 📖 **Documentation**: Help improve our docs
- 🧪 **Testing**: Test on different platforms

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google MediaPipe**: Exceptional pose detection technology
- **Google Gemini AI**: Powerful AI for dynamic feedback generation
- **Murf AI**: High-quality voice synthesis
- **Streamlit**: Beautiful and intuitive web framework
- **OpenCV**: Robust computer vision capabilities
- **The Fitness Community**: Inspiration for AI-powered coaching

## 🚀 Future Roadmap

- 🏃 **Multi-Exercise Support**: Squats, push-ups, planks, and more
- 🏆 **Gamification**: Achievement badges, streaks, and challenges
- 📱 **Mobile App**: Native iOS and Android applications
- 🤝 **Social Features**: Share workouts and compete with friends
- 🎯 **Workout Plans**: AI-generated personalized training programs
- 🩺 **Health Integration**: Heart rate monitoring and recovery tracking
- 🌐 **Cloud Sync**: Cross-device workout history and progress

---

**Ready to revolutionize your fitness journey? Get started with VoiceGym today!** 🏋️💪

For support, questions, or feature requests, please [open an issue](https://github.com/Balasubramanyam-Chilukala/VoiceGym/issues) or reach out to the development team.
