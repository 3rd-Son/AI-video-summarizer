import os
from moviepy.editor import VideoFileClip
from langchain.llms import OpenAI
from dotenv import load_dotenv
import tempfile
import assemblyai as aai
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI()


# Function to extract audio from a video
def extract_audio_from_video(video_path):
    with VideoFileClip(video_path) as video:
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        video.audio.write_audiofile(temp_audio_file.name)
        return temp_audio_file.name


# Function to transcribe audio using AssemblyAI
def transcribe_audio_assemblyai(audio_path):
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_path)
    return transcription.text


# Function to summarize text using OpenAI
def summarize_with_openai(text: str):
    messages = [
        ("system", "You are a helpful assistant that summarizes text."),
        ("human", f"Summarize the following text concisely:\n\n{text}"),
    ]
    return llm.invoke(messages).content


# Main function to process the video and generate a summary
def process_video_to_summary(video_path):
    print("Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path)

    print("Transcribing audio...")
    transcription = transcribe_audio_assemblyai(audio_path)

    print("Summarizing transcription...")
    summary = summarize_with_openai(transcription)

    print("---------------------------------------------")
    return summary


if __name__ == "__main__":
    video_path = "/Users/appleplay/Desktop/video summary/text_video.mp4"
    summary = process_video_to_summary(video_path)
    print("Summary:", summary)
