import os
import librosa
import numpy as np
import json
import spacy
from panns_inference import AudioTagging, labels
from sentence_transformers import SentenceTransformer, util
from subject_verb_object import *

audio_folder = './data/test_audio_output'
json_file_path = './data/test_captions.json'

# Parameters
num_clips = 5  # Number of audio clips to process
sr = 32000  # Sample rate
clip_duration = 5  # Audio segment length
overlap = 2  # Overlap in audio (s)
weight_audio_conf = 0.5  # Audio confidence weight
weight_text_sim = 0.5  # Text similarity weight
top_n_tags = 3  # Take the top 3 tags regardless of confidence

# Pre-trained model for sentence embeddings (SBERT)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
at = AudioTagging(checkpoint_path=None, device='cpu')

# Load AudioSet ontology labels
audio_set_labels = labels  # Assuming labels are predefined

# Function to compute similarity between caption words and AudioSet tags
def compute_similarity(caption_words, audio_set_labels):
    if not caption_words:
        return []

    # Compute embeddings for caption words and AudioSet labels
    caption_embeddings = sbert_model.encode(caption_words, batch_size=32, show_progress_bar=False)
    audio_embeddings = sbert_model.encode(audio_set_labels, batch_size=32, show_progress_bar=False)

    # Compute cosine similarity between captions and AudioSet labels
    similarity_matrix = util.cos_sim(caption_embeddings, audio_embeddings).numpy()

    confidence_scores = []
    for i, caption_word in enumerate(caption_words):
        # Find the most similar AudioSet label
        best_match_idx = np.argmax(similarity_matrix[i])
        similarity_score = similarity_matrix[i][best_match_idx]

        # If the similarity is high enough, add the AudioSet label
        if similarity_score > 0.5:  # Set threshold for match
            confidence_scores.append((audio_set_labels[best_match_idx], similarity_score))

    return confidence_scores

# Extract SVO (Subject-Verb-Object) triples and related phrases from audio captions
def extract_svo_from_captions(captions_data):
    nlp_svo = spacy.load('en_core_web_trf')  # Using transformer model for better accuracy

    connected_phrases = {}

    for key, captions in list(captions_data.items())[:num_clips]:
        audio_captions = captions.get("audio_captions", [])  # Only use audio captions

        connected_phrases[key] = {"svo_triples": [], "noun_phrases": [], "individual_words": []}

        for caption in audio_captions:
            doc = nlp_svo(caption)
            svos = findSVAOs(doc)  # Extract Subject-Verb-Object triples

            noun_phrases = set()
            individual_words = set()

            for chunk in doc.noun_chunks:
                noun_phrases.add(chunk.text)

            for token in doc:
                if token.pos_ in {"NOUN", "VERB"}:
                    individual_words.add(token.text)

            connected_phrases[key]["svo_triples"].extend(svos)
            connected_phrases[key]["noun_phrases"] = list(noun_phrases)
            connected_phrases[key]["individual_words"] = list(individual_words)

    return connected_phrases

# Process audio and extract tags
def process_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    segment_length = int(clip_duration * sr)
    hop_length = int((clip_duration - overlap) * sr)

    tags_list = []
    clipwise_confidences = {}

    for start_idx in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start_idx : start_idx + segment_length][None, :]
        clipwise_output, _ = at.inference(segment)

        sorted_indices = np.argsort(clipwise_output[0])[::-1][:top_n_tags]
        tags = [(labels[i], clipwise_output[0][i]) for i in sorted_indices]

        # Store segment tags but do not print them
        tags_list.append(tags)

        # Track max confidence for each tag over all segments
        for label, confidence in tags:
            if label not in clipwise_confidences or confidence > clipwise_confidences[label]:
                clipwise_confidences[label] = confidence

    # Convert dictionary to sorted list based on confidence
    clipwise_tags = sorted(clipwise_confidences.items(), key=lambda x: x[1], reverse=True)

    return tags_list, clipwise_tags

if __name__ == "__main__":
    with open(json_file_path, 'r') as f:
        captions_data = json.load(f)

    connected_phrases = extract_svo_from_captions(captions_data)  # Extract SVO triples and phrases

    all_tags = []
    processed_clips = 0

    # Process the audio files
    for file_name in sorted(os.listdir(audio_folder)):
        if processed_clips >= num_clips:
            break
        if file_name.endswith('.wav'):
            audio_path = os.path.join(audio_folder, file_name)
            segment_tags, clipwise_tags = process_audio(audio_path)
            all_tags.append((segment_tags, clipwise_tags))  # Store both segment-wise and clip-level tags
            processed_clips += 1

    # Clip processing
    for i, (segment_tags, clipwise_tags) in enumerate(all_tags):
        file_name = sorted(os.listdir(audio_folder))[i]  # Get the current file name
        file_number = file_name.split('.')[0]  # Remove the '.wav' extension and get the number
        
        key = list(connected_phrases.keys())[i] if i < len(connected_phrases) else None
        if key:
            caption_words = (
                connected_phrases[key]["noun_phrases"] 
                + connected_phrases[key]["individual_words"]
            )

            print(f"\n*")
            print(f"SVO Triples: {connected_phrases[key]['svo_triples']}")
            print(f"Extracted Words: {', '.join(caption_words)}")
            print(f"*")

            # Align caption words with AudioSet ontology
            aligned_tags = compute_similarity(caption_words, audio_set_labels)
            
            print("-" * 80)
            print(f"{'Tag':<30} {'Confidence Score':<15}")
            print("-" * 80)

            # Display aligned tags
            for tag, conf in aligned_tags:
                print(f"{tag:<30} {conf:.2f}")

            print("-" * 80)
