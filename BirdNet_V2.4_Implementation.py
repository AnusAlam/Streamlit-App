import streamlit as st
import birdnet
import time

st.title("BirdNet_V2.4")
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()

    st.audio(audio_bytes, format="audio/wav")
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.caption("Model starts analyzing...")

    model = birdnet.load("acoustic", "2.4", "tf")

    result = model.predict("temp_audio.wav")

    df = result.to_dataframe()
    filtered_df = df.loc[[df['confidence'].idxmax()]]

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.caption(f'Processing {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.04)

    st.header(filtered_df['species_name'])



