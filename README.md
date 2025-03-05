# StemAI - AI-Powered Audio Stem Separator

StemAI is a Streamlit web application that allows users to upload an audio file and automatically separate it into four different stems: Drums, Bass, Melody, and Vocals using the Demucs deep learning model. The app also provides an analysis of the energy distribution among the extracted stems and allows users to download each separated stem.

# Features
🎤 Audio Separation: Uses Demucs to extract drums, bass, melody, and vocals from an audio file.
📊 Stem Analysis: Calculates and visualizes the contribution of each stem based on RMS energy.
🎧 Audio Playback: Allows users to listen to each separated stem within the app.
📥 Download Stems: Provides a download button for each extracted stem.

# Usage Guide
1️⃣ Upload an .mp3, .wav, or .flac file.
2️⃣ Wait for processing to complete.
3️⃣ View & Analyze the bar chart of stem energy distribution.
4️⃣ Play the extracted stems directly in the app.
5️⃣ Download individual stems as .wav files.


To run the code, enter command: streamlit StemAI.py

# Dependencies
- torch
- librosa
- demucs
- soundfile
- numpy
- matplotlib
- seaborn
- streamlit
