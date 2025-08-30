# VoiceGym

**AI-Powered Fitness Coaching with Computer Vision and Voice Feedback**

A revolutionary fitness application that combines computer vision, artificial intelligence, and multilingual voice coaching to provide real-time exercise guidance, form analysis, and personalized feedback.

## Features

### Core Functionality
- **Multi-Exercise Detection**: Automatically detects 8+ exercises including bicep curls, push-ups, squats, lunges, shoulder press, plank, jumping jacks, and mountain climbers
- **Real-Time Form Analysis**: Advanced biomechanical analysis with injury prevention warnings
- **Intelligent Rep Counting**: Accurate repetition tracking using pose estimation
- **AI-Powered Feedback**: Context-aware feedback coaching using Google Gemini AI

### Advanced Features
- **Multilingual Support**: Text-to-speech in multiple languages (Hindi, Chinese, French, German, Spanish, English)
- **Progress Tracking**: Comprehensive workout analytics and progress visualization
- **Achievement System**: Unlockable badges and milestone tracking
- **Global Challenges**: Community-driven fitness challenges
- **Streak Tracking**: Daily workout consistency monitoring
- **Social Features**: Leaderboards and family group challenges

### Technical Highlights
- **Enterprise Database**: PostgreSQL with JSONB storage for flexible data management
- **Real-Time Processing**: Optimized computer vision pipeline with MediaPipe
- **Audio Pipeline**: Advanced text-to-speech with caching and fallback systems
- **Scalable Architecture**: Modular design with connection pooling and concurrent processing

## Technology Stack

- **Computer Vision**: MediaPipe, OpenCV
- **AI/ML**: Google Gemini AI (gemini-1.5-flash), NumPy
- **Database**: PostgreSQL with psycopg2
- **Audio**: Murf SDK/API for TTS, Pygame for playback
- **Backend**: Python with threading and concurrent processing

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Webcam for pose detection
- API keys for Gemini AI and Murf TTS

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voicegym.git
   cd voicegym
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL database**
   
   **For Linux (Ubuntu/Debian):**
   ```bash
   # Install PostgreSQL
   sudo apt-get install postgresql postgresql-contrib
   
   # Create database
   sudo -u postgres psql
   CREATE DATABASE voicegym;
   CREATE USER voiceuser WITH PASSWORD 'yourpassword';
   GRANT ALL PRIVILEGES ON DATABASE voicegym TO voiceuser;
   \q
   ```

   **For Windows:**
   ```powershell
   # Download and install PostgreSQL from https://www.postgresql.org/download/windows/
   # During installation, remember the password you set for the 'postgres' user
   
   # Open Command Prompt or PowerShell as Administrator
   # Navigate to PostgreSQL bin directory (usually):
   cd "C:\Program Files\PostgreSQL\15\bin"
   
   # Connect to PostgreSQL
   psql -U postgres
   
   # Create database and user
   CREATE DATABASE voicegym;
   CREATE USER voiceuser WITH PASSWORD 'yourpassword';
   GRANT ALL PRIVILEGES ON DATABASE voicegym TO voiceuser;
   \q
   ```

   **Alternative Windows Method (using pgAdmin):**
   1. Open pgAdmin (installed with PostgreSQL)
   2. Connect to your PostgreSQL server
   3. Right-click on "Databases" → "Create" → "Database"
   4. Name: `voicegym`
   5. Right-click on "Login/Group Roles" → "Create" → "Login/Group Role"
   6. Name: `voiceuser`, set password, grant privileges

4. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   # API Keys
   GEMINI_API_KEY=your_gemini_api_key_here
   MURF_API_KEY=your_murf_api_key_here
   
   # Database Configuration
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=voicegym
   POSTGRES_USER=voiceuser
   POSTGRES_PASSWORD=yourpassword
   ```

5. **Run the application**
   ```bash
   python temp.py
   ```

## Usage

### Controls
- **P**: Pause/Resume workout
- **Q**: Quit application
- **S**: Save current workout to database
- **R**: Show real-time statistics
- **1-6**: Manual exercise switching
- **Auto-detection**: System automatically detects exercise changes

### Getting Started
1. Launch the application
2. Select your preferred voice coach (multiple languages available)
3. Position yourself in front of the camera
4. Begin exercising - the system will automatically detect your exercise type
5. Follow real-time voice feedback for form corrections
6. View your progress and achievements

## API Keys Setup

### Google Gemini AI
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` file as `GEMINI_API_KEY`

### Murf API
1. Sign up at [Murf.ai](https://murf.ai/)
2. Navigate to API section in dashboard
3. Generate API key
4. Add to `.env` file as `MURF_API_KEY`

## Database Schema

The application uses PostgreSQL with advanced features:
- **Users**: Profile and statistics tracking
- **Workouts**: Detailed session records with JSONB exercise data
- **Achievements**: Gamification system
- **Global Challenges**: Community features
- **Exercise Records**: Personal bests and performance history

## Supported Exercises

1. **Bicep Curls**: Arm strengthening with elbow stability analysis
2. **Push-ups**: Full-body exercise with form alignment checking
3. **Squats**: Lower body strength with depth and knee alignment analysis
4. **Lunges**: Balance and leg strength assessment
5. **Shoulder Press**: Overhead movement with shoulder safety monitoring
6. **Plank**: Core stability with body alignment analysis
7. **Jumping Jacks**: Cardio exercise with coordination tracking
8. **Mountain Climbers**: High-intensity core and cardio exercise

## Voice Coaches Available

- **Hindi (India)**: Rahul, Amit, Kabir, Shweta, Ayushi
- **Chinese (Mandarin)**: Zhang, Tao, Jiao, Wei
- **French (France)**: Maxime, Louis, Adélie, Justine
- **English (US)**: Ken, Ryan, Natalie, Samantha
- **German (Germany)**: Available voices
- **Spanish (Spain)**: Available voices

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Input  │───▶│  Pose Detection │───▶│ Exercise Logic  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Output  │◀───│  TTS Pipeline   │◀───│   AI Feedback   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Progress UI   │◀───│   Database      │◀───│ Session Manager │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Performance Considerations

- **Frame Rate**: Optimized for 30 FPS real-time processing
- **Memory Usage**: Efficient landmark processing with history buffers
- **Network**: API calls are cached and rate-limited
- **Database**: Connection pooling for concurrent users
- **Audio**: Asynchronous TTS generation with queue management

## Troubleshooting

### Common Issues

1. **Camera not detected**
   - Ensure webcam is connected and not used by other applications
   - Check camera permissions
   - **Windows**: Check Device Manager for camera drivers
   - **Linux**: Verify camera access with `ls /dev/video*`

2. **Database connection failed**
   
   **Linux:**
   ```bash
   # Check if PostgreSQL is running
   sudo service postgresql status
   # Start if not running
   sudo service postgresql start
   ```
   
   **Windows:**
   ```powershell
   # Check PostgreSQL service status
   Get-Service postgresql*
   # Start service if stopped
   Start-Service postgresql-x64-15  # Replace with your version
   
   # Alternative: Use Services.msc GUI
   # Press Win+R, type "services.msc", find PostgreSQL service
   ```
   
   - Check database credentials in `.env` file
   - Ensure database and user exist

3. **Python package installation issues**
   
   **Windows specific:**
   ```powershell
   # If pip install fails, try:
   python -m pip install --upgrade pip
   # For MediaPipe issues on Windows:
   pip install mediapipe --no-deps
   pip install opencv-python
   # For audio issues:
   pip install pygame --upgrade
   ```
   
   **Linux:**
   ```bash
   # Install system dependencies for OpenCV
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   # For audio dependencies
   sudo apt-get install portaudio19-dev python3-pyaudio
   ```

4. **API quota exceeded**
   - System includes intelligent fallbacks
   - Check API usage in respective dashboards
   - Consider upgrading API plans for extended use

5. **Poor exercise detection**
   - Ensure good lighting conditions
   - Stand 3-6 feet from camera
   - Wear contrasting colors for better pose detection
   - **Windows**: Disable Windows camera app if running

6. **Audio not playing**
   
   **Windows:**
   ```powershell
   # Check if audio service is running
   Get-Service | Where-Object {$_.Name -like "*audio*"}
   # Restart Windows Audio service if needed
   Restart-Service -Name "AudioSrv"
   ```
   
   **Linux:**
   ```bash
   # Check audio system
   pulseaudio --check -v
   # Install additional audio dependencies if needed
   sudo apt-get install alsa-utils pulseaudio
   ```
   
   - Check system audio settings
   - Verify pygame audio initialization
   - Ensure audio files are being generated



## Contact

**Developer**: Balasubramanyam Chilukala
**Project Link**: [https://github.com/Balasubramanyam-Chilukala/VoiceGym](https://github.com/Balasubramanyam-Chilukala/VoiceGym)

---
