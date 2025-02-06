import os
from moviepy.editor import VideoFileClip

video_folder = "./data/test_videos"
output_folder = "./data/audio_output"

for filename in os.listdir(video_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(video_folder, filename)
        
        # Define output audio path (replace .mp4 with .wav)
        audio_filename = os.path.splitext(filename)[0] + ".wav"
        audio_path = os.path.join(output_folder, audio_filename)

        # Extract audio
        print(f"Processing {filename}...")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
        clip.close()

        print(f"Audio saved to {audio_path}")