#   ytsum - YouTube Video AI Summarizer 🎥

`ytsum` is an AI-powered YouTube video summarization tool that extracts key information from YouTube videos, transcribes audio, generates summary, and provides an interactive wordcloud visualization.
## 📌 Features

- **Video Downloading**: Download YouTube videos in high quality.
- **Audio Extraction**: Extract audio from the downloaded videos.
- **Transcription**: Automatically transcribe video audio into text using Whisper.
- **Summarization**: Generate concise summaries using advanced AI models.
- **Word Cloud**: Generate a word cloud from the transcript to visualize important keywords.

## ⚙️ Installation

### Requirements

- Python >= 3.7
- Required libraries: `streamlit`, `yt-dlp`, `whisper`, `transformers`, `pydub`, `moviepy`, `tqdm`, etc.

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### Setup
```bash
git clone https://github.com/pajay/ytsum.git
cd ytsum
```

### Usage
```bash
streamlit run app.py
```
## 🚀 Streamlit Demo
You can also use a live instance of `ytsum` on [Streamlit](https://streamlit.io/). (TO BE DEPLOYED)