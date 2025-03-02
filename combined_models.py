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
weight_audio_conf = 0.4  # Audio confidence weight
weight_caption_boost = 0.6  # Text similarity weight
top_n_tags = 3  # Take the top 3 tags regardless of confidence

# Pre-trained model for sentence embeddings (SBERT)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
at = AudioTagging(checkpoint_path=None, device='cpu')

# Load AudioSet ontology labels
audio_set_labels = labels  # Assuming labels are predefined

# Function to compute similarity between caption words and AudioSet tags
def compute_similarity_boost(caption_element, audio_tag):
    """Compute similarity between a caption element and an audio tag"""
    # Encode both texts
    caption_emb = sbert_model.encode([caption_element], show_progress_bar=False)
    tag_emb = sbert_model.encode([audio_tag], show_progress_bar=False)
    
    # Compute cosine similarity
    similarity = float(util.cos_sim(caption_emb, tag_emb)[0][0])
    return similarity

def boost_confidence(audio_tags, caption_elements):
    """Boost audio tag confidence based on caption similarities"""
    results = {}
    
    for tag, audio_conf in audio_tags:
        # Initialize with original confidence
        max_similarity = 0.0
        matching_elements = []
        
        # Compare tag with each caption element
        for element in caption_elements:
            similarity = compute_similarity_boost(element, tag)
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity > 0.5:  # Track strong matches
                matching_elements.append((element, similarity))
        
        # Calculate boost based on best similarity
        if max_similarity > 0.5:
            boost = max_similarity * weight_caption_boost
            boosted_conf = min(audio_conf + boost, 1.0)
        else:
            boosted_conf = audio_conf
        
        results[tag] = {
            'original': audio_conf,
            'boosted': boosted_conf,
            'max_similarity': max_similarity,
            'matching_elements': matching_elements
        }
    
    return results

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

    connected_phrases = extract_svo_from_captions(captions_data)
    all_tags = []
    processed_clips = 0

    # Process audio files
    for file_name in sorted(os.listdir(audio_folder)):
        if processed_clips >= num_clips:
            break
        if file_name.endswith('.wav'):
            audio_path = os.path.join(audio_folder, file_name)
            segment_tags, clipwise_tags = process_audio(audio_path)
            all_tags.append((segment_tags, clipwise_tags))
            processed_clips += 1

    # Process each clip
    for i, (segment_tags, clipwise_tags) in enumerate(all_tags):
        file_name = sorted(os.listdir(audio_folder))[i]
        key = list(connected_phrases.keys())[i] if i < len(connected_phrases) else None
        
        if key:
            # Collect all caption elements
            caption_elements = (
                connected_phrases[key]["svo_triples"] +
                connected_phrases[key]["noun_phrases"] +
                connected_phrases[key]["individual_words"]
            )

            print(f"\n{'='*110}")
            print(f"File: {file_name}")
            print("\nCaption Elements:")
            print(f"SVO Triples: {connected_phrases[key]['svo_triples']}")
            print(f"Noun Phrases: {', '.join(connected_phrases[key]['noun_phrases'])}")
            print(f"Individual Words: {', '.join(connected_phrases[key]['individual_words'])}")
            
            # Compute boosted confidences
            results = boost_confidence(clipwise_tags, caption_elements)
            
            # Display results
            print(f"\n{'='*110}")
            print(f"Audio Tags with Caption-Based Confidence Boosting:")
            print(f"{'-'*110}")
            print(f"{'Tag':<30} {'Original':<10} {'Boosted':<10} {'Best Match':<40} {'Similarity':<10}")
            print(f"{'-'*110}")
            
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1]['boosted'],
                reverse=True
            )
            
            for tag, info in sorted_results:
                best_match = ""
                if info['matching_elements']:
                    best_match = max(info['matching_elements'], key=lambda x: x[1])[0]
                
                print(f"{tag:<30} {info['original']:<10.3f} {info['boosted']:<10.3f} "
                      f"{best_match[:40]:<40} {info['max_similarity']:<10.3f}")
            
            print(f"{'='*110}")