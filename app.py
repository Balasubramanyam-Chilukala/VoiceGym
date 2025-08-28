"""
VoiceGym Streamlit Web Application
==================================

Modern web interface for VoiceGym with real-time pose tracking,
voice feedback, and session visualization.
"""

import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import logging
from typing import Dict, List, Any

from voicegym_core import (
    VoiceGymCore, 
    CameraManager, 
    VoiceFeedbackManager, 
    CoachingEngine,
    validate_api_keys
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="VoiceGym - AI Personal Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-good {
        color: #28a745;
    }
    .status-warning {
        color: #ffc107;
    }
    .status-error {
        color: #dc3545;
    }
    .coaching-message {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class VoiceGymApp:
    """Main VoiceGym Streamlit application."""
    
    def __init__(self):
        """Initialize the VoiceGym app."""
        self.initialize_session_state()
        self.core = None
        self.camera = None
        self.voice = None
        self.coach = None
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'workout_active' not in st.session_state:
            st.session_state.workout_active = False
        if 'camera_initialized' not in st.session_state:
            st.session_state.camera_initialized = False
        if 'workout_data' not in st.session_state:
            st.session_state.workout_data = []
        if 'coaching_style' not in st.session_state:
            st.session_state.coaching_style = 'motivational'
        if 'voice_enabled' not in st.session_state:
            st.session_state.voice_enabled = True
        if 'camera_index' not in st.session_state:
            st.session_state.camera_index = 0
    
    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <h1 class="main-header">
            🏋️ VoiceGym - Your AI Personal Trainer
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Real-time pose tracking • Voice feedback • AI coaching • Bicep curl analysis
        </p>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # API Status
        st.sidebar.subheader("📡 API Status")
        api_validation = validate_api_keys()
        
        murf_status = "✅ Connected" if api_validation['murf'] else "❌ Not configured"
        gemini_status = "✅ Connected" if api_validation['gemini'] else "❌ Not configured"
        
        st.sidebar.markdown(f"**Murf AI:** {murf_status}")
        st.sidebar.markdown(f"**Gemini AI:** {gemini_status}")
        
        if not api_validation['murf']:
            st.sidebar.info("💡 gTTS will be used for voice synthesis")
        
        st.sidebar.markdown("---")
        
        # Coaching Configuration
        st.sidebar.subheader("👨‍🏫 Coaching Settings")
        
        coaching_style = st.sidebar.selectbox(
            "Coaching Style",
            ['motivational', 'technical', 'encouraging'],
            index=['motivational', 'technical', 'encouraging'].index(st.session_state.coaching_style),
            help="Choose your preferred coaching style"
        )
        
        if coaching_style != st.session_state.coaching_style:
            st.session_state.coaching_style = coaching_style
            if self.coach:
                self.coach.set_coaching_style(coaching_style)
        
        voice_enabled = st.sidebar.checkbox(
            "Enable Voice Feedback",
            value=st.session_state.voice_enabled,
            help="Toggle voice feedback on/off"
        )
        st.session_state.voice_enabled = voice_enabled
        
        st.sidebar.markdown("---")
        
        # Camera Configuration
        st.sidebar.subheader("🎥 Camera Settings")
        
        camera_index = st.sidebar.number_input(
            "Camera Index",
            min_value=0,
            max_value=5,
            value=st.session_state.camera_index,
            help="Camera device index (usually 0 for built-in camera)"
        )
        st.session_state.camera_index = camera_index
        
        # Camera test button
        if st.sidebar.button("📷 Test Camera"):
            self.test_camera()
        
        st.sidebar.markdown("---")
        
        # Workout Controls
        st.sidebar.subheader("🏋️ Workout Controls")
        
        if not st.session_state.workout_active:
            if st.sidebar.button("▶️ Start Workout", type="primary"):
                self.start_workout()
        else:
            if st.sidebar.button("⏹️ Stop Workout", type="secondary"):
                self.stop_workout()
            
            if st.sidebar.button("🔄 Reset Session"):
                self.reset_session()
        
        # Session Statistics
        if st.session_state.workout_active and self.core:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Live Stats")
            stats = self.core.get_session_stats()
            
            st.sidebar.metric("Reps", stats['reps'])
            st.sidebar.metric("Time", f"{stats['elapsed_minutes']:.1f} min")
            st.sidebar.metric("Rate", f"{stats['reps_per_minute']:.1f} reps/min")
            st.sidebar.metric("Stage", stats['current_stage'].title())
    
    def test_camera(self):
        """Test camera connection."""
        try:
            test_camera = CameraManager(camera_index=st.session_state.camera_index)
            ret, frame = test_camera.read_frame()
            
            if ret:
                st.sidebar.success("✅ Camera test successful!")
                st.session_state.camera_initialized = True
            else:
                st.sidebar.error("❌ Camera test failed - no frames")
                st.session_state.camera_initialized = False
            
            test_camera.release()
            
        except Exception as e:
            st.sidebar.error(f"❌ Camera test failed: {str(e)}")
            st.session_state.camera_initialized = False
    
    def start_workout(self):
        """Start a new workout session."""
        try:
            # Initialize components
            self.core = VoiceGymCore()
            self.camera = CameraManager(camera_index=st.session_state.camera_index)
            self.voice = VoiceFeedbackManager()
            self.coach = CoachingEngine()
            self.coach.set_coaching_style(st.session_state.coaching_style)
            
            st.session_state.workout_active = True
            st.session_state.workout_data = []
            st.success("🎯 Workout started! Position yourself in front of the camera.")
            
            # Welcome message
            if st.session_state.voice_enabled:
                welcome_msg = f"Welcome to VoiceGym! I'm your {st.session_state.coaching_style} trainer. Let's start your workout!"
                self.voice.speak(welcome_msg)
            
        except Exception as e:
            st.error(f"❌ Failed to start workout: {str(e)}")
            logger.error(f"Failed to start workout: {e}")
    
    def stop_workout(self):
        """Stop the current workout session."""
        st.session_state.workout_active = False
        
        if self.core:
            stats = self.core.get_session_stats()
            summary = self.coach.get_session_summary(stats) if self.coach else f"Workout completed! {stats['reps']} reps in {stats['elapsed_minutes']:.1f} minutes."
            
            st.success(f"🏁 {summary}")
            
            if st.session_state.voice_enabled and self.voice:
                self.voice.speak(summary)
        
        self.cleanup()
    
    def reset_session(self):
        """Reset the current session."""
        if self.core:
            self.core.reset_session()
            st.session_state.workout_data = []
            st.info("🔄 Session reset!")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.camera:
                self.camera.release()
            if self.core:
                self.core.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def render_main_content(self):
        """Render the main content area."""
        if not st.session_state.workout_active:
            self.render_welcome_screen()
        else:
            self.render_workout_screen()
    
    def render_welcome_screen(self):
        """Render the welcome screen when no workout is active."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 15px; margin: 20px 0;">
                <h3>🎯 Ready to Start Your Workout?</h3>
                <p>VoiceGym uses advanced AI to track your bicep curls and provide real-time feedback.</p>
                <br>
                <h4>✨ Features:</h4>
                <ul style="text-align: left; display: inline-block;">
                    <li>🎥 Real-time pose tracking with MediaPipe</li>
                    <li>🗣️ AI-powered voice coaching</li>
                    <li>📊 Live statistics and progress tracking</li>
                    <li>🏋️ Bicep curl form analysis</li>
                    <li>💪 Multiple coaching styles</li>
                </ul>
                <br>
                <p><strong>Click "Start Workout" in the sidebar to begin!</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # System requirements check
        st.subheader("🔧 System Check")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📦 Dependencies**")
            deps_status = self.check_dependencies()
            for dep, status in deps_status.items():
                icon = "✅" if status else "❌"
                st.markdown(f"{icon} {dep}")
        
        with col2:
            st.markdown("**🎥 Camera**")
            camera_status = "✅ Ready" if st.session_state.camera_initialized else "❓ Not tested"
            st.markdown(camera_status)
            st.markdown("*Use 'Test Camera' in sidebar*")
        
        with col3:
            st.markdown("**🔑 API Keys**")
            api_validation = validate_api_keys()
            for service, status in api_validation.items():
                icon = "✅" if status else "❌"
                st.markdown(f"{icon} {service.title()}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available."""
        deps = {}
        
        try:
            import cv2
            deps["OpenCV"] = True
        except ImportError:
            deps["OpenCV"] = False
        
        try:
            import mediapipe
            deps["MediaPipe"] = True
        except ImportError:
            deps["MediaPipe"] = False
        
        try:
            import pygame
            deps["Pygame"] = True
        except ImportError:
            deps["Pygame"] = False
        
        try:
            from gtts import gTTS
            deps["gTTS"] = True
        except ImportError:
            deps["gTTS"] = False
        
        return deps
    
    def render_workout_screen(self):
        """Render the main workout screen."""
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_camera_feed()
        
        with col2:
            self.render_stats_panel()
    
    def render_camera_feed(self):
        """Render the camera feed with pose detection."""
        st.subheader("🎥 Live Camera Feed")
        
        if not self.camera or not self.camera.is_opened():
            st.error("❌ Camera not available")
            return
        
        # Create placeholder for video stream
        video_placeholder = st.empty()
        feedback_placeholder = st.empty()
        
        # Process video frames
        try:
            ret, frame = self.camera.read_frame()
            
            if ret:
                # Process pose
                processed_frame, pose_data = self.core.process_pose(frame)
                processed_frame = self.core.add_overlay(processed_frame, pose_data)
                
                # Convert BGR to RGB for Streamlit
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
                # Handle feedback
                if pose_data['feedback_needed']:
                    rep_message = self.coach.get_rep_message(pose_data['reps'])
                    feedback_placeholder.success(f"✅ {rep_message}")
                    
                    if st.session_state.voice_enabled:
                        # Use threading to avoid blocking UI
                        threading.Thread(
                            target=self.voice.speak, 
                            args=(rep_message,),
                            daemon=True
                        ).start()
                
                if pose_data['coaching_needed']:
                    coaching_tip = self.coach.get_coaching_tip(pose_data['angle'], pose_data['reps'])
                    feedback_placeholder.info(f"💬 Coach: {coaching_tip}")
                    
                    if st.session_state.voice_enabled:
                        threading.Thread(
                            target=self.voice.speak,
                            args=(coaching_tip,),
                            daemon=True
                        ).start()
                
                # Store workout data
                timestamp = datetime.now()
                self.store_workout_data(timestamp, pose_data)
                
            else:
                st.error("❌ Failed to read camera frame")
                
        except Exception as e:
            st.error(f"❌ Camera processing error: {str(e)}")
            logger.error(f"Camera processing error: {e}")
    
    def store_workout_data(self, timestamp: datetime, pose_data: Dict[str, Any]):
        """Store workout data for visualization."""
        data_point = {
            'timestamp': timestamp,
            'reps': pose_data['reps'],
            'angle': pose_data['angle'],
            'stage': pose_data['stage'],
            'landmarks_detected': pose_data['landmarks_detected']
        }
        
        st.session_state.workout_data.append(data_point)
        
        # Keep only last 100 data points for performance
        if len(st.session_state.workout_data) > 100:
            st.session_state.workout_data = st.session_state.workout_data[-100:]
    
    def render_stats_panel(self):
        """Render the statistics panel."""
        st.subheader("📊 Workout Statistics")
        
        if not self.core:
            st.info("Start a workout to see statistics")
            return
        
        # Current session stats
        stats = self.core.get_session_stats()
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Reps", stats['reps'])
            st.metric("Time", f"{stats['elapsed_minutes']:.1f} min")
        
        with col2:
            st.metric("Rate", f"{stats['reps_per_minute']:.1f} reps/min")
            st.metric("Stage", stats['current_stage'].title())
        
        # Progress visualization
        if len(st.session_state.workout_data) > 1:
            self.render_progress_charts()
    
    def render_progress_charts(self):
        """Render progress charts."""
        df = pd.DataFrame(st.session_state.workout_data)
        
        if len(df) < 2:
            return
        
        # Reps over time
        st.subheader("📈 Progress Charts")
        
        # Rep progression
        fig_reps = go.Figure()
        fig_reps.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['reps'],
            mode='lines+markers',
            name='Reps',
            line=dict(color='#2E86AB', width=3)
        ))
        
        fig_reps.update_layout(
            title="Rep Count Over Time",
            xaxis_title="Time",
            yaxis_title="Total Reps",
            height=300
        )
        
        st.plotly_chart(fig_reps, use_container_width=True)
        
        # Angle visualization
        fig_angle = go.Figure()
        fig_angle.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['angle'],
            mode='lines',
            name='Arm Angle',
            line=dict(color='#F18F01', width=2)
        ))
        
        fig_angle.update_layout(
            title="Arm Angle (Recent Activity)",
            xaxis_title="Time",
            yaxis_title="Angle (degrees)",
            height=300
        )
        
        st.plotly_chart(fig_angle, use_container_width=True)
    
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
        
        # Auto-refresh for live updates
        if st.session_state.workout_active:
            time.sleep(0.1)  # Small delay for smooth updates
            st.rerun()


def main():
    """Main entry point for the Streamlit app."""
    app = VoiceGymApp()
    app.run()


if __name__ == "__main__":
    main()