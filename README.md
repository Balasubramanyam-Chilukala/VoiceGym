# VoiceGym
# 🏋️ VoiceGym: Your AI Personal Trainer

> A real-time, voice-enabled AI fitness coach that runs entirely on your local machine. It uses your webcam, computer vision, and generative AI to analyze your exercise form, count your reps, and provide live, verbal feedback.

This project turns your webcam into an intelligent personal trainer that focuses on a single exercise (bicep curls) to provide a complete, interactive feedback loop. It demonstrates the future of accessible, AI-driven home fitness without needing a web server.


## 🌟 Key Features

-   **Real-Time Pose Estimation:** Utilizes Google's MediaPipe library to track your key body points in real-time through your webcam.
-   **Live Voice Coaching:** All feedback, rep counts, and motivation are delivered as natural-sounding speech, powered by the **Murf AI API**, for a truly hands-free workout.
-   **AI-Powered Form Analysis:** The script can be easily modified to use the **Google Gemini API** to analyze your joint angles and provide instant, actionable feedback on your form. The current version uses a robust, pre-scripted feedback system.
-   **Automatic Rep Counting:** A custom logic engine, based on biomechanical angles of your arm, accurately counts your bicep curl repetitions.
-   **On-Screen Visuals:** A clean, real-time display of your rep count, current exercise stage, and live joint angles is overlaid on your camera feed using OpenCV.
-   **Asynchronous Audio:** Verbal feedback is played in a separate thread, ensuring the video feed remains smooth and never freezes while the coach is speaking.

## 🛠️ Technology Stack

| Component                | Technology / Library                                       |
| :----------------------- | :--------------------------------------------------------- |
| **Language** | Python                                                     |
| **Computer Vision** | OpenCV, MediaPipe                                          |
| **AI Feedback Engine** | Google Gemini API (or the included pre-scripted logic) |
| **Voice Synthesis (TTS)**| Murf AI API                                                |
| **Audio Playback** | Pygame                                                     |
| **Environment** | Anaconda / Conda / venv                                    |

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
    pip install opencv-python mediapipe numpy requests pygame python-dotenv google-generativeai
    ```

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
