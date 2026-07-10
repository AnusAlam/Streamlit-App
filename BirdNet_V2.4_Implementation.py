import streamlit as st
import birdnet
import time
import os

st.title("BirdNet_V2.4")

# Cache model configuration
@st.cache_resource
def load_birdnet_model():
    return birdnet.load("acoustic", "2.4", "tf")

model = load_birdnet_model()

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])


# Helper function to execute prediction safely
def analyze_audio_safely(model_obj, file_path):
    """
    By encapsulating the code logic here, we satisfy the context 
    requirements for child workers spawned by the backend.
    """
    return model_obj.predict(file_path)

if uploaded_file is not None:
    file_name = uploaded_file.name
    st.subheader(f"*{file_name}*")
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")
    
    temp_file_path = os.path.abspath("temp_audio.wav")
    with open(temp_file_path, "wb") as f:
        f.write(audio_bytes)

    st.caption("Model starts analyzing")

    try:
        # Execute through the safe function wrapper
        result = analyze_audio_safely(model, temp_file_path)
        df = result.to_dataframe()
        
        if not df.empty:
            filtered_df = df.loc[[df['confidence'].idxmax()]]

            # Progress bar animation
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(100):
                latest_iteration.caption(f'Processing {i+1}%')
                bar.progress(i + 1)
                time.sleep(0.01)

            st.header(f":blue[*{filtered_df['species_name'].values[0]}*]")
        else:
            st.warning("No bird species could be confidently identified.")
            
    except Exception as e:
        st.error(f"An error occurred in model prediction: {e}")
