import os
import librosa
import numpy as np
import json
import spacy
from panns_inference import AudioTagging, labels
from sentence_transformers import SentenceTransformer, util

audio_folder = './data/test_audio_output'
json_file_path = './data/test_captions.json'

# Parameters
num_clips = 5  # Number of audio clips to process
sr = 32000  # Sample rate
clip_duration = 5  # Audio segment length
overlap = 3  # Overlap in audio (s)
weight_audio_conf = 0.8  # Audio confidence weight
weight_text_sim = 0.2  # Text similarity weight
top_n_tags = 3  # Take the top 3 tags regardless of confidence

nlp = spacy.load('en_core_web_sm')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
at = AudioTagging(checkpoint_path=None, device='cpu')

# Process audio and extract tags
def process_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    segment_length = int(clip_duration * sr)
    hop_length = int((clip_duration - overlap) * sr)

    tags_list = []
    for start_idx in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start_idx : start_idx + segment_length][None, :]
        clipwise_output, _ = at.inference(segment)

        # Get top 3 tags instead of using a threshold
        sorted_indices = np.argsort(clipwise_output[0])[::-1][:top_n_tags]
        tags = [(labels[i], clipwise_output[0][i]) for i in sorted_indices]
        tags_list.append(tags)

    return tags_list

def extract_noun_verb_phrases(captions_data):
    connected_phrases = {}

    for key, captions in list(captions_data.items())[:num_clips]:
        all_captions = captions.get("audio_captions", []) + captions.get("visual_captions", []) + captions.get("audio_visual_captions", []) + captions.get("GPT_AV_captions", [])

        connected_phrases[key] = {"noun_verb_pairs": [], "compound_nouns": []}

        for caption in all_captions:
            doc = nlp(caption)
            noun_verb_pairs = set()
            compound_nouns = set()
            
            for token in doc:
                if token.pos_ == "VERB":
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"): 
                            phrase = f"{child.text} {token.text}"
                            noun_verb_pairs.add(phrase)

                if token.dep_ == "compound" and token.head.pos_ == "NOUN":
                    phrase = f"{token.text} {token.head.text}"
                    compound_nouns.add(phrase)

            connected_phrases[key]["noun_verb_pairs"] = list(noun_verb_pairs)
            connected_phrases[key]["compound_nouns"] = list(compound_nouns)

    return connected_phrases

# Similarity between audio tags and text captions
def compute_similarity(audio_tags, caption_words):
    if not caption_words:
        return [(tag, conf, conf * weight_audio_conf) for tag, conf in audio_tags]  # Reduced confidence without text
    
    if not audio_tags:
        return [] 

    audio_labels = [tag for tag, _ in audio_tags]
    audio_embeddings = sbert_model.encode(audio_labels, batch_size=32, show_progress_bar=False) 
    text_embeddings = sbert_model.encode(caption_words, batch_size=32, show_progress_bar=False)

    if text_embeddings.shape[0] == 0:  # If no valid text embeddings
        return [(tag, conf, conf * weight_audio_conf) for tag, conf in audio_tags]

    similarity_matrix = util.cos_sim(audio_embeddings, text_embeddings).numpy()
    
    confidence_scores = []
    for i, (tag, conf) in enumerate(audio_tags):
        max_sim = np.max(similarity_matrix[i]) if similarity_matrix.shape[1] > 0 else 0
        final_conf = weight_audio_conf * conf + weight_text_sim * max_sim
        confidence_scores.append((tag, conf, final_conf))  # Store both original and adjusted confidence
    
    return confidence_scores

if __name__ == "__main__":
    with open(json_file_path, 'r') as f:
        captions_data = json.load(f)

    all_tags = []
    processed_clips = 0
    for file_name in sorted(os.listdir(audio_folder)):
        if processed_clips >= num_clips:
            break
        if file_name.endswith('.wav'):
            audio_path = os.path.join(audio_folder, file_name)
            tags = process_audio(audio_path)
            all_tags.append(tags)
            processed_clips += 1

    connected_phrases = extract_noun_verb_phrases(captions_data)

    # Clip processing
    for i, clip_tags in enumerate(all_tags):
        key = list(connected_phrases.keys())[i] if i < len(connected_phrases) else None
        if key:
            caption_words = connected_phrases[key]["noun_verb_pairs"] + connected_phrases[key]["compound_nouns"]

            print(f"\nAudio Clip {i + 1} - Caption Words: {', '.join(caption_words)}")
            print("-" * 60)
            print(f"{'Segment':<10} {'Tag':<30} {'Audio Conf.':<15} {'Final Conf.':<15}")
            print("-" * 60)

            for j, segment_tags in enumerate(clip_tags):
                final_scores = compute_similarity(segment_tags, caption_words)

                if final_scores:
                    for tag, audio_conf, final_conf in final_scores:
                        print(f"{j + 1:<10} {tag:<30} {audio_conf:.2f}         {final_conf:.2f}")

            print("-" * 60)
