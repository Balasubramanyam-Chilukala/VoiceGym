import streamlit as st
import cv2
import numpy as np
import time
from threading import Thread
import tempfile
import os
from voicegym import VoiceGymLocal, speak_feedback, test_audio, AUDIO_AVAILABLE, logger
import logging

# Set page config
st.set_page_config(
    page_title="🏋️ VoiceGym Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitVoiceGym:
    def __init__(self):
        self.gym = None
        self.camera_running = False
        self.current_camera = 0
        
    def get_available_cameras(self):
        """Get list of available camera indices."""
        available_cameras = []
        for i in range(5):  # Check first 5 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(i)
                cap.release()
            except Exception as e:
                logger.warning(f"Camera {i} check failed: {e}")
        return available_cameras
    
    def init_camera_preview(self, camera_index):
        """Initialize camera preview."""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return None, "Camera not available"
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test frame capture
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None, "Cannot read from camera"
            
            return cap, "Camera ready"
            
        except Exception as e:
            return None, f"Camera error: {str(e)}"
    
    def process_camera_frame(self, frame):
        """Process frame using VoiceGym logic."""
        if self.gym:
            return self.gym.process_frame(frame)
        return frame

# Initialize session state
if 'streamlit_gym' not in st.session_state:
    st.session_state.streamlit_gym = StreamlitVoiceGym()

st.title("🏋️ VoiceGym Coach - Streamlit Interface")
st.markdown("### Your AI Personal Trainer with Voice Feedback")

# Sidebar for controls
st.sidebar.header("🎛️ Controls")

# Audio system status
st.sidebar.subheader("🔊 Audio System")
if AUDIO_AVAILABLE:
    st.sidebar.success("✅ Audio system available")
    if st.sidebar.button("🔊 Test Audio"):
        with st.spinner("Testing audio..."):
            result = test_audio()
            if result:
                st.sidebar.success("✅ Audio test successful!")
            else:
                st.sidebar.error("❌ Audio test failed")
else:
    st.sidebar.warning("🔇 Audio system not available")
    st.sidebar.info("Voice feedback will be shown as text only")

# Voice feedback test
st.sidebar.subheader("🗣️ Voice Feedback Test")
test_text = st.sidebar.text_input("Test message:", "Welcome to VoiceGym!")
if st.sidebar.button("🔊 Test Voice"):
    with st.spinner("Generating voice..."):
        result = speak_feedback(test_text)
        if result:
            st.sidebar.success("✅ Voice generation successful!")
        else:
            st.sidebar.warning("⚠️ Voice generation failed or running in silent mode")

# Camera selection
st.sidebar.subheader("📷 Camera Setup")
available_cameras = st.session_state.streamlit_gym.get_available_cameras()

if available_cameras:
    camera_options = {f"Camera {i}": i for i in available_cameras}
    selected_camera_name = st.sidebar.selectbox(
        "Select Camera:",
        options=list(camera_options.keys()),
        index=0
    )
    selected_camera = camera_options[selected_camera_name]
    st.session_state.streamlit_gym.current_camera = selected_camera
    
    st.sidebar.success(f"✅ {len(available_cameras)} camera(s) detected")
else:
    st.sidebar.error("❌ No cameras detected")
    st.error("No cameras found. Please check your camera connections.")
    st.stop()

# Camera preview section
st.header("📹 Camera Preview")

col1, col2 = st.columns([2, 1])

with col1:
    # Camera preview placeholder
    camera_placeholder = st.empty()
    
with col2:
    st.subheader("📊 Workout Stats")
    stats_placeholder = st.empty()
    
    st.subheader("🎯 Controls")
    
    # Preview mode toggle
    preview_mode = st.checkbox("📹 Camera Preview", value=True)
    
    # Workout mode toggle
    workout_mode = st.checkbox("🏋️ Start Workout", value=False)
    
    # Reset button
    if st.button("🔄 Reset Workout"):
        if st.session_state.streamlit_gym.gym:
            st.session_state.streamlit_gym.gym.reps = 0
            st.session_state.streamlit_gym.gym.stage = "ready"
        st.success("Workout reset!")

# Camera operation
if preview_mode or workout_mode:
    cap, status = st.session_state.streamlit_gym.init_camera_preview(
        st.session_state.streamlit_gym.current_camera
    )
    
    if cap is None:
        st.error(f"Camera Error: {status}")
    else:
        # Initialize VoiceGym if in workout mode and not already initialized
        if workout_mode and st.session_state.streamlit_gym.gym is None:
            try:
                # Create VoiceGym instance with selected camera
                st.session_state.streamlit_gym.gym = VoiceGymLocal(
                    camera_index=st.session_state.streamlit_gym.current_camera
                )
                st.success("🏋️ VoiceGym initialized! Start your bicep curls!")
                
                # Initial welcome message
                welcome_msg = "Welcome to VoiceGym! Position yourself in front of the camera and start your workout!"
                speak_feedback(welcome_msg)
                
            except Exception as e:
                st.error(f"Failed to initialize VoiceGym: {e}")
                workout_mode = False
        
        # Camera stream
        frame_placeholder = camera_placeholder.container()
        
        # Read and display frames
        try:
            ret, frame = cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame if in workout mode
                if workout_mode and st.session_state.streamlit_gym.gym:
                    processed_frame = st.session_state.streamlit_gym.process_camera_frame(frame)
                    
                    # Update stats
                    with stats_placeholder.container():
                        st.metric("💪 Reps", st.session_state.streamlit_gym.gym.reps)
                        st.metric("📐 Current Stage", st.session_state.streamlit_gym.gym.stage)
                        
                        # Show last feedback time
                        current_time = time.time()
                        time_since_speech = current_time - st.session_state.streamlit_gym.gym.last_speech
                        if time_since_speech < 10:
                            st.info(f"🔊 Voice cooldown: {10-time_since_speech:.1f}s")
                else:
                    processed_frame = frame
                    
                    with stats_placeholder.container():
                        st.info("📹 Preview mode - Enable workout to start tracking")
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(
                    frame_rgb,
                    caption=f"Camera {st.session_state.streamlit_gym.current_camera}",
                    use_column_width=True
                )
                
            else:
                st.error("Failed to read camera frame")
                
        except Exception as e:
            st.error(f"Camera processing error: {e}")
        finally:
            cap.release()

# Instructions section
st.header("📖 Instructions")

instructions_col1, instructions_col2 = st.columns(2)

with instructions_col1:
    st.subheader("🎯 How to Use")
    st.markdown("""
    1. **Select your camera** from the dropdown in the sidebar
    2. **Enable camera preview** to see yourself
    3. **Test audio** to ensure voice feedback works
    4. **Start workout mode** to begin pose tracking
    5. **Position yourself** so your left arm is visible
    6. **Start doing bicep curls** and get real-time feedback!
    """)

with instructions_col2:
    st.subheader("🔧 Troubleshooting")
    st.markdown("""
    - **No camera detected**: Check camera connections and permissions
    - **Audio not working**: Audio system will run in silent mode
    - **Voice feedback delayed**: Normal cooldown period between messages
    - **Pose not detected**: Ensure good lighting and full arm visibility
    - **Reset if needed**: Use the reset button to start over
    """)

# System information
st.header("💻 System Information")
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric("📷 Available Cameras", len(available_cameras))

with info_col2:
    audio_status = "✅ Available" if AUDIO_AVAILABLE else "❌ Not Available"
    st.metric("🔊 Audio System", audio_status)

with info_col3:
    workout_status = "🏋️ Active" if workout_mode else "📹 Preview"
    st.metric("🎯 Mode", workout_status)

# Footer
st.markdown("---")
st.markdown("**🏋️ VoiceGym Coach** - Your AI Personal Trainer with Voice Feedback")