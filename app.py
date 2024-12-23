import logging
import os
import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import moviepy.editor as mp
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torch
import whisper
import yt_dlp
from PIL import Image
from accelerate import Accelerator
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Step 1: Download video from YouTube
def download_video_from_youtube(video_url, output_path):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_path, 'downloaded_video.%(ext)s'),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            for file in os.listdir(output_path):
                if file.startswith('downloaded_video'):
                    return os.path.join(output_path, file)
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        raise


# Step 2: Extract audio from video and save as MP4
def extract_audio_from_video(video_path, audio_path, audio_mp4_path):
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.audio.write_audiofile(audio_mp4_path, codec="aac")
        video.close()
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        raise


# Step 3: Split audio into chunks based on transcribed text length (target 700-800 words per chunk)
def split_audio_into_chunks(audio_path, chunk_length_ms=60000):  # Default 1-minute chunks
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    start_ms = 0
    total_length_ms = len(audio)

    while start_ms < total_length_ms:
        end_ms = min(start_ms + chunk_length_ms, total_length_ms)
        chunk = audio[start_ms:end_ms]
        chunk_path = f"{audio_path}_chunk_{start_ms // 60000}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        start_ms = end_ms  # Move to the next chunk

    return chunks


# Step 4: Convert audio chunk to text using Whisper model
def convert_audio_to_text(audio_path):
    try:
        model = whisper.load_model("base")  # Load the Whisper model (choose based on your system)
        result = model.transcribe(audio_path)

        # Check if transcription was successful
        if "text" not in result:
            raise ValueError(f"Transcription failed: {result}.")

        text = result["text"]
        return text
    except Exception as e:
        logging.error(f"Error converting audio to text: {e}")
        raise


# Step 5: Generate a top-line summary using knkarthick/MEETING_SUMMARY for extractive summarization
def summarize_text(accelerator, model, tokenizer, text, summary_type):
    try:
        if len(text.strip()) == 0:
            raise ValueError("Text is empty. Cannot summarize.")

        # Tokenize input text and prepare it for summarization
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

        # Move the input tensors to the appropriate device
        inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

        # Generate the summary based on the selected summary type
        if summary_type == 1:  # Short summary
            summary_ids = model.generate(**inputs, max_length=100, min_length=40, num_beams=4, early_stopping=True)
        elif summary_type == 2:  # Medium summary
            summary_ids = model.generate(**inputs, max_length=175, min_length=85, num_beams=4, early_stopping=True)
        elif summary_type == 3:  # Detailed summary
            summary_ids = model.generate(**inputs, max_length=350, min_length=175, num_beams=4, early_stopping=True)
        else:
            raise ValueError(f"Invalid summary type: {summary_type}.")

        # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return None


# Create an interactive word cloud with Plotly
def generate_interactive_wordcloud(text):
    # Generate the word cloud with WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Convert wordcloud to a list of words and frequencies
    word_freq = wordcloud.words_

    # Convert to a DataFrame for easier manipulation with Plotly
    word_freq_data = [{'word': word, 'frequency': freq} for word, freq in word_freq.items()]

    # Create the interactive word cloud with Plotly
    df = pd.DataFrame(word_freq_data)
    fig = px.scatter(df, x='word', y='frequency', size='frequency', text='word',
                     title="Interactive Word Cloud", labels={'word': 'Word', 'frequency': 'Frequency'},
                     template="plotly_dark")
    fig.update_traces(marker=dict(sizemode='diameter', opacity=0.6, line=dict(width=0.5, color='white')))
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


# Create a 3D word cloud with Plotly
def generate_3d_wordcloud(text):
    # Generate the word cloud with WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Convert wordcloud to a list of words and frequencies
    word_freq = wordcloud.words_

    # Convert to a DataFrame for easier manipulation with Plotly
    word_freq_data = [{'word': word, 'frequency': freq} for word, freq in word_freq.items()]
    df = pd.DataFrame(word_freq_data)

    # Normalize frequencies to map them to a range for 3D positioning
    df['z'] = np.log(df['frequency'] * 1000)  # Log transform to avoid extreme Z-values

    # Create random positions for words in 3D space (X, Y, Z)
    np.random.seed(42)  # For reproducibility
    df['x'] = np.random.uniform(-10, 10, size=len(df))
    df['y'] = np.random.uniform(-10, 10, size=len(df))

    # Create the 3D scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z', text='word', size='frequency',
                        title="3D Interactive Word Cloud",
                        labels={'word': 'Word', 'frequency': 'Frequency', 'z': 'Frequency (log scale)'},
                        template="plotly_dark")

    fig.update_traces(marker=dict(sizemode='diameter', opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


# Cleanup previous downloaded files (chunks, video, audio)
def cleanup_previous_files(output_path):
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove individual files
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
            except Exception as e:
                logging.error(f"Error cleaning up {file_path}: {e}")


# Process each chunk
def process_chunk(accelerator, model, tokenizer, chunk_path, summary_type):
    try:
        # Step 4: Convert chunk audio to text
        transcribed_text = convert_audio_to_text(chunk_path)

        if transcribed_text:
            # Step 5: Summarize the transcribed text
            summary = summarize_text(accelerator, model, tokenizer, transcribed_text, summary_type)
            return transcribed_text, summary
        else:
            return "", ""
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_path}: {e}")
        return "", ""


# Main function to handle Streamlit app logic
# Main function to handle Streamlit app logic
def main():
    # Initialize Accelerator
    accelerator = Accelerator()

    # Load pre-trained model and tokenizer from knkarthick/MEETING_SUMMARY
    model_name = "knkarthick/MEETING_SUMMARY"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare the model and tokenizer for the Accelerator
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Streamlit app UI
    st.title("ytsum - Youtube Summarizer")
    video_url = st.text_input("Enter YouTube Video URL:")
    summary_type = st.slider("Select Summary Type:", 1, 3, 2, format="Summary Length: %d")
    analyze_button = st.button("Analyze Video")

    # Check if results already exist in session state
    if "full_transcript" in st.session_state and "final_summary" in st.session_state:
        full_transcript = st.session_state["full_transcript"]
        final_summary = st.session_state["final_summary"]
    else:
        full_transcript = ""
        final_summary = ""

    # Sidebar for metadata and thumbnail
    st.sidebar.header("Video Metadata")
    video_metadata = None
    thumbnail = None

    if video_url:
        # Retrieve video metadata
        try:
            video_metadata = download_video_from_youtube(video_url, "downloaded_videos")
            video_data = yt_dlp.YoutubeDL().extract_info(video_url, download=False)
            thumbnail_url = video_data.get('thumbnail')

            # Download the thumbnail image
            response = requests.get(thumbnail_url)
            if response.status_code == 200:
                thumbnail = Image.open(BytesIO(response.content))
                st.sidebar.image(thumbnail, caption="Video Thumbnail", use_container_width=True)
            else:
                st.sidebar.text("Error: Unable to fetch thumbnail.")

            st.sidebar.text(f"Title: {video_data.get('title')}")
            st.sidebar.text(f"Published: {video_data.get('upload_date')}")
            st.sidebar.text(f"Duration: {video_data.get('duration')} sec")
            st.sidebar.text(f"Views: {video_data.get('view_count')}")
        except Exception as e:
            st.sidebar.text("Error fetching video data.")
            st.sidebar.text(f"Error: {e}")

    if analyze_button:
        st.subheader("Goblin drums far away...")
        progress_bar = st.progress(0)

        # Clean up previous files
        cleanup_previous_files("downloaded_videos")

        # Process the video
        try:
            video_path = download_video_from_youtube(video_url, "downloaded_videos")
            audio_path = os.path.join("downloaded_videos", "extracted_audio.wav")
            audio_mp4_path = os.path.join("downloaded_videos", "extracted_audio.mp4")

            extract_audio_from_video(video_path, audio_path, audio_mp4_path)
            chunks = split_audio_into_chunks(audio_path)

            all_text = ""
            full_transcript = ""  # Variable to store the full transcript

            # Process chunks with progress bar
            with ThreadPoolExecutor() as executor:
                for i, result in enumerate(tqdm(
                        executor.map(
                            lambda chunk_path: process_chunk(accelerator, model, tokenizer, chunk_path, summary_type),
                            chunks), desc="Processing Chunks", total=len(chunks))):
                    progress_bar.progress((i + 1) / len(chunks))
                    transcribed_text, summary = result
                    all_text += transcribed_text + " "
                    full_transcript += transcribed_text + "\n"  # Add each chunk's transcription to the full transcript

            # Store the full transcript in session state
            st.session_state["full_transcript"] = full_transcript

            # Display the full transcript
            st.subheader("Full Transcript:")
            st.text_area("Transcript", full_transcript, height=400)  # Adjust the height as needed

            # Final summary output
            st.subheader("Final Summary of the Video:")
            final_summary = summarize_text(accelerator, model, tokenizer, all_text, summary_type)

            # Store the final summary in session state
            st.session_state["final_summary"] = final_summary

            # Display final summary in a larger, resizable text area
            st.text_area("Summary", final_summary, height=400)  # Adjust the height parameter to make the window larger

            # Generate the interactive 3D word cloud
            st.subheader("Word Cloud")
            wordcloud_3d = generate_3d_wordcloud(all_text)
            st.plotly_chart(wordcloud_3d)

        except Exception as e:
            st.error(f"Error: {e}")

    # Only show the download button if the transcript is available and processing is complete
    if "full_transcript" in st.session_state and st.session_state["full_transcript"]:
        st.download_button(
            label="Download Transcript",
            data=st.session_state["full_transcript"],
            file_name="transcript.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
