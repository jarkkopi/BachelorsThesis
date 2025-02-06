import os
import librosa
import numpy as np
import json
import spacy
from panns_inference import AudioTagging, labels

audio_folder = './data/test_audio_output'

at = AudioTagging(checkpoint_path=None, device='cuda')

sr = 32000 
clip_duration = 5
overlap = 2
threshold = 0.5

# process audio and extract tags
def process_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    segment_length = int(clip_duration * sr)
    hop_length = int((clip_duration - overlap) * sr)

    # tags for each segment
    tags_list = []
    for start_idx in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start_idx : start_idx + segment_length][None, :]  # (batch_size, segment_samples)

        clipwise_output, _ = at.inference(segment)
        tags = [labels[i] for i, score in enumerate(clipwise_output[0]) if score > threshold]

        tags_list.append(tags)

    return tags_list

# first 10 audio clips
all_tags = []
processed_clips = 0
for file_name in sorted(os.listdir(audio_folder)):
    if processed_clips >= 10:
        break

    if file_name.endswith('.wav'):
        audio_path = os.path.join(audio_folder, file_name)
        tags = process_audio(audio_path)
        all_tags.append(tags)
        processed_clips += 1

# results
for i, clip_tags in enumerate(all_tags):
    print(f"Audio Clip {i + 1} Tags:")
    for j, segment_tags in enumerate(clip_tags):
        print(f"  Segment {j + 1}: {segment_tags}")
    print()