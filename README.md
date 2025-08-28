# 🏋️ VoiceGym: Your AI Personal Trainer

> A real-time, voice-enabled AI fitness coach that runs entirely on your local machine. It uses your webcam, computer vision, and generative AI to analyze your exercise form, count your reps, and provide live, verbal feedback.

VoiceGym offers both a **modern web interface** and an **enhanced command-line version**, giving you flexibility in how you want to train. The modular architecture ensures reliable performance with fallback options for voice synthesis.

## 🗂️ Available Applications

### 🌐 Web Interface (`app.py`)
Modern Streamlit web application with:
- Real-time camera feed with pose visualization
- Interactive statistics and progress charts
- Configuration panel for coaching styles
- Session management and data visualization

### 💻 Enhanced CLI (`voicegym_enhanced.py`)  
Feature-rich command-line interface with:
- Multiple coaching styles (motivational, technical, encouraging)
- Robust error handling and logging
- Keyboard shortcuts for session control
- Voice feedback with gTTS fallback

### 🔧 Core Module (`voicegym_core.py`)
Shared functionality including:
- Camera management with error handling
- Pose detection and exercise tracking
- Voice synthesis with fallback options
- Coaching engine with multiple styles

## 🌟 Key Features

-   **🎥 Real-Time Pose Estimation:** Utilizes Google's MediaPipe library to track your key body points in real-time through your webcam.
-   **🗣️ Smart Voice Coaching:** Premium voice synthesis with **Murf AI API** and **gTTS fallback** ensures you always get audio feedback.
-   **🤖 AI-Powered Form Analysis:** Uses **Google Gemini API** for intelligent coaching feedback with pre-scripted alternatives.
-   **📊 Automatic Rep Counting:** Advanced biomechanical angle analysis accurately counts your bicep curl repetitions.
-   **💻 Modern Web Interface:** Beautiful Streamlit web app with real-time statistics and progress visualization.
-   **⚡ Enhanced CLI Version:** Feature-rich command-line interface with multiple coaching styles and robust error handling.
-   **🔧 Modular Architecture:** Core functionality separated for easy maintenance and testing.
-   **🛡️ Fallback Options:** gTTS backup for voice and pre-scripted coaching ensure the app always works.

## 🛠️ Technology Stack

| Component                | Technology / Library                                       |
| :----------------------- | :--------------------------------------------------------- |
| **Language** | Python 3.8+                                               |
| **Computer Vision** | OpenCV, MediaPipe                                          |
| **Web Interface** | Streamlit, Plotly, Pandas                                 |
| **AI Feedback Engine** | Google Gemini API + pre-scripted fallback              |
| **Voice Synthesis** | Murf AI API + gTTS fallback                               |
| **Audio Playback** | Pygame                                                     |
| **Environment** | python-dotenv, Virtual environments                       |

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

You will need the following software installed:
-   Python (version 3.8 - 3.10 recommended)
-   An environment manager like Anaconda/Miniconda or `venv`.

### ⚙️ Installation & Setup

1.  **Clone or Download the Repository:**
    Get the Python script and save it in a new project folder.

2.  **Set Up a Clean Environment:**
    It is highly recommended to use a virtual environment.

    *Using Conda:*
    ```bash
    # Create a new environment
    conda create --name voicegym-env python=3.9 -y
    # Activate the environment
    conda activate voicegym-env
    ```

3.  **Install Required Libraries:**
    With your environment active, install all the necessary packages with `pip`.
    ```bash
    pip install -r requirements.txt
    ```

## 🆕 New Application Structure

VoiceGym now includes multiple applications for different use cases:

### 🌐 Web Interface (`streamlit run app.py`)
- Modern Streamlit web application
- Interactive camera feed with pose visualization  
- Real-time statistics and progress charts
- Configuration panel for coaching styles
- Session management and data visualization

### 💻 Enhanced CLI (`python voicegym_enhanced.py`)
- Feature-rich command-line interface
- Multiple coaching styles (motivational, technical, encouraging)
- Robust error handling and logging
- Keyboard shortcuts for session control
- Voice feedback with gTTS fallback

### 🔧 Core Module (`voicegym_core.py`)
- Shared functionality between applications
- Camera management with error handling
- Pose detection and exercise tracking
- Voice synthesis with fallback options
- Coaching engine with multiple styles

### 🔑 Configuration

To use the AI voice and feedback features, you must configure your API keys.

1.  Create a file named `.env` in the same folder as your Python script.
2.  Add your secret API keys to this file in the following format:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    MURF_API_KEY="YOUR_MURF_API_KEY_HERE"
    ```
3.  Open the main Python script and find the following line. Replace the placeholder with a real Voice ID from your Murf AI account (e.g., `"en-US-terrell"`).
    ```python
    # In the speak_feedback function
    "voiceId": "en-US-terrell",
    ```

## ▶️ How to Run

1.  Ensure your virtual environment is active (e.g., `conda activate voicegym-env`).
2.  Run the application from your terminal:
    ```bash
    python your_script_name.py
    ```
3.  A window will open with your webcam feed. Position yourself so the camera can see you and begin performing bicep curls.
4.  To stop the program, make sure the webcam window is active and press the **'q'** key or the **ESC** key.
