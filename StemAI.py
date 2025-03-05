import os
import numpy as np
import torch
import librosa
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from demucs import pretrained
from demucs.apply import apply_model

# Step 1: Function to separate stems using Demucs
def separate_stems(input_audio, output_directory):
    #Separate an audio file into stems using Demucs
    model = pretrained.get_model(name='htdemucs')
    model.to('cpu')

    # Load audio properly
    audio, sr = librosa.load(input_audio, sr=44100, mono=False)
    audio_tensor = torch.tensor(audio).unsqueeze(0)

    # Apply the model to extract stems
    stems = apply_model(model, audio_tensor, device='cpu')

    # Save stems
    stem_names = ["drums", "bass", "melody", "vocals"]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    stem_paths = {}
    for i, stem in enumerate(stems[0]):
        stem_audio = stem.numpy()
        stem_path = os.path.join(output_directory, f"{stem_names[i]}.wav")
        sf.write(stem_path, stem_audio.T, sr)
        stem_paths[stem_names[i]] = stem_path

    return stem_paths

# Step 2: Function to calculate RMS energy
def calculate_energy(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return np.sqrt(np.mean(y**2))

# Step 3: Function to analyze stem distribution
def analyze_stems(stem_paths):
    energies = {stem: calculate_energy(path) for stem, path in stem_paths.items()}
    total_energy = sum(energies.values())
    percentages = {stem: (energy / total_energy) * 100 for stem, energy in energies.items()}
    return percentages

# Step 4: Streamlit UI
st.title("StemAI 🎵")
st.write("Upload an audio file to separate it into drums, bass, vocals, and melody stems.")

# File Upload Section
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac"])
output_directory = "output_stems"

if uploaded_file is not None:
    # Save uploaded file temporarily
    input_audio = os.path.join("temp_audio", uploaded_file.name)
    if not os.path.exists("temp_audio"):
        os.makedirs("temp_audio")

    with open(input_audio, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the audio file
    st.write("Processing audio file... ⏳")
    stem_paths = separate_stems(input_audio, output_directory)
    
    # Analyze stem contributions
    st.write("Analyzing stem contributions... 🔍")
    percentages = analyze_stems(stem_paths)

    # Display Horizontal Bar Chart using Seaborn
    st.write("### Stem Distribution")
    sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    stems, values = zip(*sorted_percentages)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=values, y=stems, palette="muted", hue=stems, legend=False)
    plt.xlabel("Percentage of Total Energy (%)")
    plt.ylabel("Stem Type")
    plt.title("Stem Contribution Distribution")
    plt.grid(False)
    ax.set_facecolor('none') 
    sns.despine()
    
    # Add percentage values to each bar
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}%", va='center')
    
    st.pyplot(plt)

    # Play each separated stem with labels
    st.write("### Play Separated Stems")
    for stem, path in stem_paths.items():
        st.write(f"**{stem.capitalize()} Stem:**")
        st.audio(path, format="audio/wav", start_time=0)
    # Add a download button for each stem
        st.download_button(
            label=f"Download",
            data=open(path, "rb").read(),
            file_name=f"{stem}.wav",
            mime="audio/wav"
        )