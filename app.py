"""
ytsum - YouTube Summarizer Streamlit App

This application allows users to input a YouTube video URL, select summarization options,
and generate a transcript and summary of the video. It also includes a 3D word cloud
visualization and allows users to download the transcript and summary.

Features:
- YouTube URL input for video analysis
- Text summarization with various options
- Downloadable transcript and summary
- 3D word cloud visualization
"""

import asyncio
import json
import logging
import os
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import whisper
from accelerate import Accelerator
from moviepy import editor as mp
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import detect_nonsilent
from playwright.sync_api import sync_playwright
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from wordcloud import WordCloud
import yt_dlp


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_cookies_from_browser():
    """
    Collects cookies from a browser session using Selenium in headless mode.

    Returns:
        list: A list of cookies to be used with yt-dlp.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.youtube.com")

        # Wait for YouTube to load cookies
        page.wait_for_timeout(5000)

        # Collect cookies
        cookies = context.cookies()
        browser.close()

        # Convert cookies to yt-dlp compatible format
        cookies_dict = {cookie["name"]: cookie["value"] for cookie in cookies}

        # Save cookies to a file
        cookie_file = "cookies.txt"
        with open(cookie_file, "w") as f:
            json.dump(cookies_dict, f)

        return cookie_file


def download_video(video_url, output_path):
    """
    Downloads a video from a given URL using yt-dlp and saves it to the specified output path.

    Args:
        video_url (str): The URL of the video to download.
        output_path (str): The directory path where the video will be saved.

    Returns:
        str or None: The file path of the downloaded video if successful, otherwise None.

    Raises:
        yt_dlp.DownloadError: If an error occurs during the download process.
        Exception: If any other unexpected error occurs.
    """
    try:
        cookie_file = get_cookies_from_browser()
        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": os.path.join(output_path, "downloaded_video.%(ext)s"),
            "cookies": cookie_file,
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            },
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            return next(
                (
                    os.path.join(output_path, file)
                    for file in os.listdir(output_path)
                    if file.startswith("downloaded_video")
                ),
                None,
            )
    except Exception as e:
        logging.error("Error downloading video: %s", e)
        raise


def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file and saves it to a specified path.

    Args:
        video_path (str): The file path to the input video.
        audio_path (str): The file path where the extracted audio will be saved.

    Logs:
        Info: Logs the start and successful completion of the audio extraction process.
        Error: Logs any errors encountered during the extraction process.
    """
    try:
        logging.info("Starting audio extraction.")
        with mp.VideoFileClip(video_path) as video:
            audio = video.audio
            audio.write_audiofile(audio_path)
        logging.info("Audio extraction is successful.")
    except (OSError, ValueError) as e:
        logging.error("Error extracting audio: %s", e)
    except Exception as e:
        logging.error("Unexpected error extracting audio: %s", e)


def split_audio(
    audio_path,
    chunk_length_ms=60000,
    silence_threshold=-50.0,
    min_silence_len=100,
    max_workers=4,
):
    """
    Splits an audio file into smaller chunks based on specified parameters.

    Parameters:
        audio_path (str): Path to the input WAV audio file.
        chunk_length_ms (int, optional): Length of each chunk in ms (default: 60000).
        silence_threshold (float, optional): Silence threshold in dBFS (default: -50.0).
        min_silence_len (int, optional): Minimum silence length in milliseconds to
        consider a split (default: 100).
        max_workers (int, optional): Maximum threads for processing (default: 4).

    Returns:
        list: Paths to the exported audio chunks.

    Logs errors if the audio cannot be decoded or other exceptions occur.
    """

    try:
        audio = AudioSegment.from_wav(audio_path)
        total_length_ms = len(audio)
        chunks = []

        def is_valid_chunk(chunk):
            nonsilent_segments = detect_nonsilent(
                chunk, min_silence_len, silence_threshold
            )
            return len(nonsilent_segments) > 0

        def export_chunk(start, end):
            chunk = audio[start:end]
            if is_valid_chunk(chunk):
                chunk_path = f"{os.path.splitext(audio_path)[0]}_chunk_{start // chunk_length_ms}.wav"
                chunk.export(chunk_path, format="wav")
                return chunk_path
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    export_chunk,
                    start_ms,
                    min(start_ms + chunk_length_ms, total_length_ms),
                )
                for start_ms in range(0, total_length_ms, chunk_length_ms)
            ]
            for task in tasks:
                result = task.result()
                if result:
                    chunks.append(result)
        return chunks
    except CouldntDecodeError as e:
        logging.error("Error decoding audio: %s - %s", audio_path, e)
        return []
    except (ValueError, OSError, TypeError) as e:
        logging.error("Error splitting audio into chunks: %s", e)
        return []
    except Exception as e:
        logging.error("Unexpected error splitting audio into chunks: %s", e)
        return []


def transcribe_audio(audio_path):
    """
    Transcribes audio from a given file path using the Whisper model.

    This function loads an audio file, processes it into a mel spectrogram,
    and uses a pre-trained Whisper model to transcribe the audio into text.
    It handles errors by logging them and returns an empty string if the
    audio is invalid or an error occurs.

    Args:
        audio_path (str): The file path to the audio file to be transcribed.

    Returns:
        str: The transcribed text from the audio file, or an empty string if
        transcription fails.
    """
    try:
        model = whisper.load_model("base")
        audio = whisper.audio.load_audio(audio_path)
        if audio.shape[0] == 0 or audio is None:
            logging.error("Empty or invalid audio tensor: %s", audio_path)
            return ""
        audio = whisper.audio.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        result = whisper.decode(
            model, mel, whisper.DecodingOptions(fp16=torch.cuda.is_available())
        )
        return result.text
    except (ValueError, OSError) as e:
        logging.error("Error transcribing audio: %s - %s", audio_path, e)
        return ""
    except Exception as e:
        logging.error("Unexpected error transcribing audio: %s - %s", audio_path, e)
        return ""


def summarize_text(
    accelerator,
    model,
    tokenizer,
    text,
    summary_type,
    use_sampling,
    top_k,
    top_p,
    temperature,
):
    """
    Summarizes the given text using a specified model and tokenizer.

    Parameters:
        accelerator (Accelerator): The device accelerator for model execution.
        model (AutoModelForSeq2SeqLM): The pre-trained model for text summarization.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        text (str): The input text to be summarized.
        summary_type (int): The type of summary to generate, determining length.
        use_sampling (bool): Whether to use sampling for text generation.
        top_k (int): The number of highest probability vocabulary tokens to keep for sampling.
        top_p (float): The cumulative probability for nucleus sampling.
        temperature (float): The temperature for sampling, affecting randomness.

    Returns:
        str: The summarized text or None if an error occurs.

    Raises:
        ValueError: If the input text is empty or an invalid summary type is provided.
    """
    try:
        if not text.strip():
            raise ValueError("Text is empty. Cannot summarize.")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=min(1024, len(text.split())),
            padding=True,
        ).to(accelerator.device)

        summary_params = {
            1: {"max_length": 100, "min_length": 40},
            2: {"max_length": 250, "min_length": 100},
            3: {"max_length": 600, "min_length": 250},
        }

        if summary_type not in summary_params:
            raise ValueError(f"Invalid summary type: {summary_type}")

        params = summary_params[summary_type]
        if use_sampling:
            params.update(
                {
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "do_sample": True,
                }
            )
        else:
            params.update({"num_beams": 6, "early_stopping": True, "do_sample": False})

        with torch.no_grad():
            summary_ids = model.generate(**inputs, **params)

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except (ValueError, TypeError, KeyError, OSError) as e:
        logging.error("Error summarizing text: %s", e)
        return None

    except Exception as e:
        logging.error("Unexpected error summarizing text: %s", e)
        return None


def transcribe_and_summarize(
    accelerator,
    model,
    tokenizer,
    chunk_path,
    summary_type,
    use_sampling,
    top_k,
    top_p,
    temperature,
    max_workers=4,
):
    """
    Transcribes audio from a specified path and generates a summarized
    version of the transcribed text.

    The audio at `chunk_path` is transcribed using `transcribe_audio`.
    The transcribed text is split into chunks,
    summarized using `summarize_text`, and customized with parameters
    like summary type and sampling options.

    Parameters:
        accelerator (Accelerator): The device accelerator (e.g., GPU or CPU) for model execution.
        model (AutoModelForSeq2SeqLM): The model used for text summarization.
        tokenizer (AutoTokenizer): The tokenizer for text encoding and decoding.
        chunk_path (str): The file path to the audio for transcription.
        summary_type (int): The type of summary to generate.
        use_sampling (bool): Whether to use sampling methods for text generation.
        top_k (int): The number of tokens to keep for top-k filtering.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        temperature (float): Controls randomness of predictions during sampling.
        max_workers (int, optional): Maximum number of threads for processing (default: 4).

    Returns:
        tuple: Contains the transcribed text and the summarized text.

    Logs:
        - Logs warnings if the transcription is empty.
        - Logs info when transcription and summarization are completed.
        - Logs errors if any issues arise during processing.
    """

    try:
        transcribed_text = transcribe_audio(chunk_path)
        if not transcribed_text or transcribed_text.strip() == "":
            logging.warning(f"Empty transcription for chunk: {chunk_path}")
            return "", ""

        logging.info(f"Transcription completed for chunk: {chunk_path}")

        max_input_length = tokenizer.model_max_length
        tokens = tokenizer.encode(transcribed_text, truncation=False)
        chunks = [
            tokens[i : i + max_input_length]
            for i in range(0, len(tokens), max_input_length)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    summarize_text,
                    accelerator,
                    model,
                    tokenizer,
                    tokenizer.decode(chunk_tokens, skip_special_tokens=True),
                    summary_type,
                    use_sampling,
                    top_k,
                    top_p,
                    temperature,
                ): chunk_tokens
                for chunk_tokens in chunks
            }
            summarized_chunks = []
            for future in futures:
                try:
                    summary = future.result()
                    summarized_chunks.append(summary)
                    logging.info(f"Summarization completed for chunk")
                except Exception as e:
                    logging.error(f"Error summarizing chunk: {e}")
                    summarized_chunks.append("")

        final_summary = " ".join(summarized_chunks)

        return transcribed_text, final_summary

    except (OSError, ValueError, TypeError) as e:
        logging.error(f"Error transcribing and summarizing chunk {chunk_path}: {e}")
        return "", ""
    except Exception as e:
        logging.error(
            f"Unexpected error transcribing and summarizing chunk {chunk_path}: {e}"
        )
        return "", ""


def generate_wordcloud(text):
    """
    Generates a 3D scatter plot word cloud from the provided text.

    The function calculates word frequencies from the input text and visualizes them in a 3D scatter plot using Plotly,
    where word size and color reflect frequency.

    Parameters:
        text (str): The input text for generating the word cloud.

    Returns:
        plotly.graph_objs._figure.Figure: A 3D scatter plot representing the word cloud.
    """

    wordcloud = WordCloud(
        width=2000, height=800, background_color="white", max_words=200
    ).generate(text)
    word_freq = wordcloud.words_
    df = pd.DataFrame(
        [{"word": word, "frequency": freq} for word, freq in word_freq.items()]
    )
    df["z"] = np.log(df["frequency"] * 1000)
    df["x"], df["y"] = np.linspace(-5, 5, len(df)), np.linspace(-5, 5, len(df))
    np.random.shuffle(df["y"].values)
    df["size"] = df["frequency"] * 1000
    df["color"] = df["frequency"]
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        text="word",
        size="size",
        color="color",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        scene={
            "xaxis": {"showgrid": False},
            "yaxis": {"showgrid": False},
            "zaxis": {"showgrid": False},
        }
    )
    return fig


def clean_up_files(output_path):
    """
    Remove all files and directories within the specified output path.

    This function recursively traverses the specified directory and removes all files,
    subdirectories, and their contents. It ensures that all items within the directory
    are deleted, leaving the directory itself empty.

    Parameters:
        output_path (str): The path to the directory whose contents will be deleted.
                            The directory itself will remain intact after the operation.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        OSError: If there is an error while removing files or directories.
    """

    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def fetch_video_metadata(video_url):
    """
    Fetches metadata for a given video URL using yt-dlp.

    Args:
        video_url (str): The URL of the video to fetch metadata for.

    Returns:
        tuple: A tuple containing the video's title, uploader, duration, description,
        thumbnail URL, video ID, upload date, view count, like count, dislike count,
        category, and tags. If an error occurs, returns default values for each field.
    """
    try:
        with yt_dlp.YoutubeDL() as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            title = info_dict.get("title", "Unknown Title")
            uploader = info_dict.get("uploader", "Unknown Uploader")
            duration = info_dict.get("duration", 0)
            description = info_dict.get("description", "No description available.")
            thumbnail_url = info_dict.get("thumbnail", "")
            video_id = info_dict.get("id", "Unknown Video ID")
            upload_date = info_dict.get("upload_date", "Unknown Date")
            view_count = info_dict.get("view_count", "Unknown Views")
            like_count = info_dict.get("like_count", "Unknown Likes")
            dislike_count = info_dict.get("dislike_count", "Unknown Dislikes")
            category = info_dict.get("category", "Unknown Category")
            tags = info_dict.get("tags", [])
            return (
                title,
                uploader,
                duration,
                description,
                thumbnail_url,
                video_id,
                upload_date,
                view_count,
                like_count,
                dislike_count,
                category,
                tags,
            )
    except (
        OSError,
        ValueError,
        KeyError,
    ) as e:
        logging.error("Error fetching video metadata: %s", e)
    return "", "", 0, "", "", "", "", "", "", "", []


def main():
    """
    Main function for the 'ytsum - YouTube Summarizer' Streamlit app.

    Sets up the interface for users to input a YouTube URL, select summarization options,
    and view results including the transcript, summary, and 3D word cloud.
    Users can also download the transcript and summary.

    Note:
        - Requires a YouTube video URL input.
        - Uses Streamlit components for interaction and display.
    """

    st.set_page_config(page_title="ytsum")
    st.title("ytsum - YouTube Summarizer")

    video_url = st.text_input("Enter YouTube Video URL:")
    summary_type = st.slider(
        "Select Summary Type:", 1, 3, 2, format="Summary Length: %d"
    )
    analyze_button = st.button("Analyze Video")

    with st.expander("Advanced Options"):
        use_sampling = st.checkbox(
            "Use Sampling options - For more diverse summarization", value=False
        )
        top_k = st.slider("Top-k", 1, 200, 20) if use_sampling else 20
        top_p = st.slider("Top-p", 0.0, 2.0, 0.7) if use_sampling else 0.7
        temperature = st.slider("Temperature", 0.0, 5.0, 0.5) if use_sampling else 0.5

    # Initialize components
    accelerator = Accelerator()
    model_name = "knkarthick/MEETING_SUMMARY"

    # Create a Streamlit progress bar
    st.write(
        "<small>Model and Tokenizer Initialization: </small>", unsafe_allow_html=True
    )
    st.write(f"<small>{model_name}</small>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    # Track progress with tqdm
    with tqdm(total=2, desc="Loading model and tokenizer", dynamic_ncols=True) as pbar:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pbar.update(1)
        progress_bar.progress(50)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pbar.update(1)
        progress_bar.progress(100)

    # Prepare model and tokenizer using Accelerator
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Validate tokenizer
    if not callable(tokenizer):
        raise TypeError(
            "Tokenizer is not callable. Ensure it was initialized correctly."
        )

    logging.debug(f"Tokenizer type: {type(tokenizer)}")

    if analyze_button and video_url:
        with st.spinner("Goblin drums far away..."):
            try:
                # Fetch video metadata
                (
                    title,
                    uploader,
                    duration,
                    description,
                    thumbnail_url,
                    video_id,
                    upload_date,
                    view_count,
                    like_count,
                    dislike_count,
                    category,
                    tags,
                ) = fetch_video_metadata(video_url)

                # Display video metadata on sidebar
                st.sidebar.header("Video Metadata")
                st.sidebar.image(thumbnail_url, use_container_width=True)
                st.sidebar.write(f"**Title**: {title}")
                st.sidebar.write(f"**Uploader**: {uploader}")
                st.sidebar.write(f"**Duration**: {duration} seconds")
                st.sidebar.write(f"**Video ID**: {video_id}")
                st.sidebar.write(f"**Upload Date**: {upload_date}")
                st.sidebar.write(f"**View Count**: {view_count}")
                st.sidebar.write(f"**Like Count**: {like_count}")
                st.sidebar.write(f"**Dislike Count**: {dislike_count}")
                st.sidebar.write(f"**Category**: {category}")
                st.sidebar.write(
                    f"**Tags**: {', '.join(tags) if tags else 'No tags available'}"
                )
                st.sidebar.write(f"**Description**: {description[:300]}...")

                st.sidebar.markdown(
                    """
                        ---
                        Created by [th3pajay](https://github.com/th3pajay)
                        ![UserGIF](https://user-images.githubusercontent.com/74038190/219925470-37670a3b-c3e2-4af7-b468-673c6dd99d16.png)
                    """
                )

                # Download video and process audio
                clean_up_files("downloaded_videos")
                video_path = download_video(video_url, "downloaded_videos")
                audio_path = os.path.join("downloaded_videos", "extracted_audio.wav")
                extract_audio(video_path, audio_path)
                chunks = split_audio(audio_path)

                full_transcript = ""
                all_text = ""

                # Process each chunk
                with ThreadPoolExecutor() as executor:
                    results = list(
                        tqdm(
                            executor.map(
                                lambda chunk: transcribe_and_summarize(
                                    accelerator,
                                    model,
                                    tokenizer,
                                    chunk,
                                    summary_type,
                                    use_sampling,
                                    top_k,
                                    top_p,
                                    temperature,
                                ),
                                chunks,
                            ),
                            total=len(chunks),
                        )
                    )

                for transcribed_text, summary in results:
                    full_transcript += transcribed_text + "\n"
                    all_text += transcribed_text + " "

                # Display results
                st.session_state["full_transcript"] = full_transcript
                st.subheader("Full Transcript:")
                st.text_area("Transcript", full_transcript, height=400)

                # Add transcript download button
                st.download_button(
                    "Download Transcript", full_transcript, file_name="transcript.txt"
                )

                final_summary = summarize_text(
                    accelerator,
                    model,
                    tokenizer,
                    all_text,
                    summary_type,
                    use_sampling,
                    top_k,
                    top_p,
                    temperature,
                )
                st.session_state["final_summary"] = final_summary

                st.subheader("Final Summary:")
                st.text_area("Summary", final_summary, height=400)

                # Add summary download button
                st.download_button(
                    "Download Summary", final_summary, file_name="summary.txt"
                )

                wordcloud_3d = generate_wordcloud(all_text)
                st.subheader("3D Word Cloud")
                st.plotly_chart(wordcloud_3d)

            except (
                OSError,
                ValueError,
                FileNotFoundError,
            ) as e:
                logging.error("Error processing video: %s", e)
                st.error(f"Error processing video: {e}")
            except Exception as e:
                logging.error("Unexpected error processing video: %s", e)
                st.error(f"Unexpected error processing video: {e}")


if __name__ == "__main__":
    main()
