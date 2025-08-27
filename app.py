"""
VoiceGym Streamlit App - Enhanced UI for AI-powered fitness coaching
"""

import streamlit as st
import cv2
import asyncio
import threading
import time
import queue
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from voicegym_core import VoiceGymCore
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🏋️ VoiceGym: AI Personal Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .workout-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stats-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'voicegym_core' not in st.session_state:
        st.session_state.voicegym_core = None
    if 'workout_active' not in st.session_state:
        st.session_state.workout_active = False
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'feedback_queue' not in st.session_state:
        st.session_state.feedback_queue = queue.Queue()
    if 'workout_history' not in st.session_state:
        st.session_state.workout_history = []

def get_available_voices():
    """Get available Murf voices (this would be expanded with actual API call)."""
    return {
        "en-US-terrell": "Terrell (English US - Male)",
        "en-US-jenny": "Jenny (English US - Female)", 
        "en-GB-oliver": "Oliver (English UK - Male)",
        "en-GB-emma": "Emma (English UK - Female)",
        "en-AU-ryan": "Ryan (English AU - Male)",
        "es-ES-pablo": "Pablo (Spanish - Male)",
        "fr-FR-claire": "Claire (French - Female)",
        "de-DE-hans": "Hans (German - Male)",
        "it-IT-sofia": "Sofia (Italian - Female)"
    }

def get_available_languages():
    """Get available languages for voice synthesis."""
    return [
        "English",
        "Spanish", 
        "French",
        "German",
        "Italian",
        "Portuguese",
        "Japanese",
        "Korean",
        "Chinese (Mandarin)"
    ]

def get_coaching_styles():
    """Get available coaching styles."""
    return {
        "Gentle/Encouraging": "Supportive and patient guidance with positive reinforcement",
        "Motivational": "Energetic and uplifting coaching to boost performance", 
        "High-intensity": "Direct and challenging coaching for maximum effort"
    }

def sidebar_configuration():
    """Render sidebar configuration options."""
    st.sidebar.header("🔧 Configuration")
    
    # API Keys Section
    st.sidebar.subheader("🔑 API Keys")
    gemini_key = st.sidebar.text_input(
        "Gemini AI API Key", 
        type="password",
        help="Get your key from https://makersuite.google.com/app/apikey"
    )
    murf_key = st.sidebar.text_input(
        "Murf AI API Key", 
        type="password",
        help="Get your key from https://murf.ai/api"
    )
    
    # Voice Configuration
    st.sidebar.subheader("🎤 Voice Settings")
    available_voices = get_available_voices()
    selected_voice = st.sidebar.selectbox(
        "Select Voice",
        options=list(available_voices.keys()),
        format_func=lambda x: available_voices[x],
        index=0
    )
    
    # Language Selection
    available_languages = get_available_languages()
    selected_language = st.sidebar.selectbox(
        "Language",
        options=available_languages,
        index=0
    )
    
    # Coaching Style
    st.sidebar.subheader("💪 Coaching Style")
    coaching_styles = get_coaching_styles()
    selected_style = st.sidebar.selectbox(
        "Coaching Style",
        options=list(coaching_styles.keys()),
        format_func=lambda x: f"{x}: {coaching_styles[x]}",
        index=1  # Default to Motivational
    )
    
    # Workout Settings
    st.sidebar.subheader("⚙️ Workout Settings")
    feedback_frequency = st.sidebar.slider(
        "Feedback Frequency (seconds)",
        min_value=10,
        max_value=60,
        value=20,
        help="How often to provide coaching feedback"
    )
    
    rep_feedback = st.sidebar.checkbox(
        "Rep Count Announcements",
        value=True,
        help="Announce each completed repetition"
    )
    
    return {
        'gemini_api_key': gemini_key,
        'murf_api_key': murf_key,
        'voice_id': selected_voice,
        'language': selected_language,
        'coaching_style': selected_style,
        'feedback_frequency': feedback_frequency,
        'rep_feedback': rep_feedback
    }

def validate_config(config):
    """Validate configuration and show warnings."""
    issues = []
    
    if not config['gemini_api_key']:
        issues.append("Gemini API key is required for AI-powered feedback")
    
    if not config['murf_api_key']:
        issues.append("Murf API key is required for voice synthesis")
    
    return issues

async def feedback_handler(event_type, data):
    """Handle feedback events from VoiceGym core."""
    if event_type == 'rep_completed':
        config = st.session_state.get('config', {})
        if config.get('rep_feedback', True):
            voicegym = st.session_state.voicegym_core
            
            rep_messages = [
                f"Fantastic! That's rep number {data['reps']}! Keep it up!",
                f"Excellent work! Rep {data['reps']} completed!",
                f"Great job! That's {data['reps']} reps down!",
                f"Perfect! Rep {data['reps']} in the books!",
                f"Outstanding! Rep {data['reps']} with great form!"
            ]
            
            import random
            message = random.choice(rep_messages)
            await voicegym.speak_feedback_async(message)

def camera_worker():
    """Worker function for camera processing in separate thread."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.feedback_queue.put(("error", "Cannot open camera"))
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    voicegym = st.session_state.voicegym_core
    frame_placeholder = st.session_state.get('frame_placeholder')
    
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        try:
            processed_frame, angle, reps, stage = voicegym.process_frame(
                frame, 
                lambda event, data: asyncio.run(feedback_handler(event, data))
            )
            
            # Update UI through queue
            st.session_state.feedback_queue.put((
                "frame_update", 
                {
                    'frame': processed_frame,
                    'angle': angle,
                    'reps': reps,
                    'stage': stage,
                    'stats': voicegym.get_session_stats()
                }
            ))
            
        except Exception as e:
            st.session_state.feedback_queue.put(("error", f"Frame processing error: {e}"))
    
    cap.release()

def render_workout_stats(stats):
    """Render workout statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
            <h3>🏋️ Reps</h3>
            <h2>{}</h2>
        </div>
        """.format(stats['reps']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
            <h3>⏱️ Duration</h3>
            <h2>{:.1f} min</h2>
        </div>
        """.format(stats['session_duration_minutes']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
            <h3>📊 Rate</h3>
            <h2>{:.1f} reps/min</h2>
        </div>
        """.format(stats['reps_per_minute']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card">
            <h3>🎯 Stage</h3>
            <h2>{}</h2>
        </div>
        """.format(stats['current_stage'].title()), unsafe_allow_html=True)

def render_rep_history_chart(rep_history):
    """Render rep history chart."""
    if len(rep_history) > 1:
        df = pd.DataFrame(rep_history)
        df['time_from_start'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        fig = px.line(
            df, 
            x='time_from_start', 
            y='rep_number',
            title='Rep Progress Over Time',
            labels={'time_from_start': 'Time (seconds)', 'rep_number': 'Reps Completed'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <h1 class="main-header">🏋️ VoiceGym: AI Personal Trainer</h1>
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
        Real-time AI-powered fitness coaching with voice feedback
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    config = sidebar_configuration()
    st.session_state.config = config
    
    # Validate configuration
    config_issues = validate_config(config)
    
    if config_issues:
        st.markdown("""
        <div class="alert-warning">
            <h4>⚠️ Configuration Issues</h4>
            <ul>
        """ + "".join([f"<li>{issue}</li>" for issue in config_issues]) + """
            </ul>
            <p>Some features may be limited without proper API keys.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-success">
            ✅ Configuration complete! Ready to start your workout.
        </div>
        """, unsafe_allow_html=True)
    
    # Main workout area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # Camera controls
        control_col1, control_col2 = st.columns(2)
        
        with control_col1:
            if st.button("🚀 Start Workout", disabled=st.session_state.workout_active):
                if not st.session_state.workout_active:
                    # Initialize VoiceGym core
                    st.session_state.voicegym_core = VoiceGymCore(config)
                    st.session_state.workout_active = True
                    st.session_state.camera_active = True
                    
                    # Start camera worker in separate thread
                    camera_thread = threading.Thread(target=camera_worker)
                    camera_thread.daemon = True
                    camera_thread.start()
                    
                    st.success("Workout started! Position yourself in front of the camera.")
                    st.rerun()
        
        with control_col2:
            if st.button("⏹️ Stop Workout", disabled=not st.session_state.workout_active):
                st.session_state.workout_active = False
                st.session_state.camera_active = False
                
                if st.session_state.voicegym_core:
                    # Save workout to history
                    stats = st.session_state.voicegym_core.get_session_stats()
                    workout_data = {
                        'date': datetime.now(),
                        'duration_minutes': stats['session_duration_minutes'],
                        'total_reps': stats['reps'],
                        'reps_per_minute': stats['reps_per_minute'],
                        'coaching_style': config['coaching_style']
                    }
                    st.session_state.workout_history.append(workout_data)
                    
                    st.session_state.voicegym_core.cleanup()
                    st.session_state.voicegym_core = None
                
                st.success("Workout stopped!")
                st.rerun()
        
        # Camera feed placeholder
        frame_placeholder = st.empty()
        st.session_state.frame_placeholder = frame_placeholder
        
        if st.session_state.workout_active:
            # Process feedback queue
            try:
                while True:
                    event_type, data = st.session_state.feedback_queue.get_nowait()
                    
                    if event_type == "frame_update":
                        # Display frame
                        frame_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Update stats in sidebar
                        with col2:
                            render_workout_stats(data['stats'])
                            
                            # Rep history chart
                            if len(data['stats']['rep_history']) > 1:
                                st.subheader("📈 Progress")
                                render_rep_history_chart(data['stats']['rep_history'])
                    
                    elif event_type == "error":
                        st.error(f"Error: {data}")
                        
            except queue.Empty:
                pass
            
            # Auto-refresh every 100ms when workout is active
            time.sleep(0.1)
            st.rerun()
        else:
            frame_placeholder.markdown("""
            <div class="workout-card">
                <h2>🎥 Camera Feed</h2>
                <p>Click "Start Workout" to begin your AI-powered training session</p>
                <p>Make sure your camera is connected and positioned to see your upper body</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Workout Stats")
        
        if st.session_state.workout_active and st.session_state.voicegym_core:
            # Live stats will be updated via the frame processing
            pass
        else:
            st.markdown("""
            <div class="stats-card">
                <h3>🏋️ Ready to Start</h3>
                <p>Configure your settings and start your workout to see live statistics</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Workout History
        st.subheader("📅 Workout History")
        
        if st.session_state.workout_history:
            history_df = pd.DataFrame(st.session_state.workout_history)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df = history_df.sort_values('date', ascending=False)
            
            st.dataframe(
                history_df[['date', 'total_reps', 'duration_minutes', 'reps_per_minute']],
                use_container_width=True
            )
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.workout_history = []
                st.rerun()
        else:
            st.info("No workout history yet. Complete a workout to see your progress!")

if __name__ == "__main__":
    main()