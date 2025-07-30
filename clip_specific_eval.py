import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer, util
import spacy
from collections import defaultdict

# === Paths ===
json_file_path = './data/val_captions.json'
csv_predictions_path = './MTURK/mturk_audio_tags_dynamic.csv'

# === Parameters ===
boost_alpha = 0.5
semantic_similarity_threshold = 0.5

# === Models ===
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
nlp = spacy.load('en_core_web_trf')
embedding_cache = {}

# === Load Data ===
with open(json_file_path, 'r') as f:
    captions_data = json.load(f)

predictions_df = pd.read_csv(csv_predictions_path)
predictions_df.set_index('filename', inplace=True)

# === Dummy ground truth for illustration ===
ground_truth_tags_dict = {
    "5265004457.wav": ['Conversation', 'Female speech, woman speaking', 'Male speech, man speaking', 'Speech'],
}

# === Embedding Cache ===
def get_embedding_with_cache(text):
    if text in embedding_cache:
        return embedding_cache[text]
    embedding = sbert_model.encode([text], show_progress_bar=False)
    embedding_cache[text] = embedding
    return embedding

# === Similarity Computation ===
def compute_similarity(text1, text2):
    emb1 = get_embedding_with_cache(text1)
    emb2 = get_embedding_with_cache(text2)
    return float(util.cos_sim(emb1, emb2)[0][0])

# === Confidence Boosting (No threshold logic) ===
def boost_confidence_ratio(audio_tags, caption_elements, audio_captions, alpha=0.1, caption_similarity_threshold=0.3):
    num_captions = len(audio_captions)
    results = {}
    for tag, audio_conf in audio_tags:
        match_count = sum(1 for word in caption_elements if compute_similarity(word, tag) > caption_similarity_threshold)
        match_ratio = min(1, match_count / num_captions) if num_captions > 0 else 0
        boosted_conf = min(1, alpha * match_ratio + (1 - alpha) * audio_conf)
        results[tag] = {
            'original': audio_conf,
            'boosted': boosted_conf,
            'matches': match_count,
        }
    return results

# === Phrase Extraction ===
def extract_caption_phrases(audio_captions):
    unique_phrases = set()
    used_token_idxs = set()
    for caption in audio_captions:
        doc = nlp(caption)

        # SVO Triplets
        for token in doc:
            if token.pos_ == "VERB":
                subj = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                obj = [child for child in token.children if child.dep_ in ("dobj", "attr", "prep", "pobj")]
                if subj and obj:
                    all_tokens = subj + [token] + obj
                elif subj:
                    all_tokens = subj + [token]
                elif obj:
                    all_tokens = [token] + obj
                else:
                    continue

                token_idxs = {t.i for t in all_tokens}
                if not token_idxs & used_token_idxs:
                    phrase = " ".join(
                        t.text.lower() for t in all_tokens
                        if not t.is_stop and not t.is_punct
                    ).strip()
                    if phrase:
                        unique_phrases.add(phrase)
                        used_token_idxs.update(token_idxs)

        # Noun chunks
        for chunk in doc.noun_chunks:
            token_idxs = {t.i for t in chunk}
            if not token_idxs & used_token_idxs:
                phrase = " ".join(
                    t.text.lower() for t in chunk
                    if not t.is_stop and not t.is_punct
                ).strip()
                if phrase:
                    unique_phrases.add(phrase)
                    used_token_idxs.update(token_idxs)

        # Compound nouns
        for token in doc:
            if token.dep_ == "compound" and token.head.pos_ == "NOUN":
                token_idxs = {token.i, token.head.i}
                if not token_idxs & used_token_idxs:
                    if not token.is_stop and not token.is_punct and not token.head.is_stop and not token.head.is_punct:
                        phrase = f"{token.text.lower()} {token.head.text.lower()}".strip()
                        unique_phrases.add(phrase)
                        used_token_idxs.update(token_idxs)

    return list(unique_phrases)

# === Plotting (no threshold markers) ===
def plot_boosting_results(results, file_name, alpha, caption_words, ground_truth, sim_threshold):
    tags = list(results.keys())
    original_conf = [results[tag]['original'] for tag in tags]
    boosted_conf = [results[tag]['boosted'] for tag in tags]
    colors = ['green' if boosted > orig else 'red' for orig, boosted in zip(original_conf, boosted_conf)]

    # Append (GT) to ground truth tags
    labels = []
    for tag in tags:
        labels.append(f"{tag} (GT)" if tag in ground_truth else tag)

    x = np.arange(len(tags))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, original_conf, width, label='Original', color='blue')
    plt.bar(x + width / 2, boosted_conf, width, label='Boosted', color=colors)

    plt.xticks(x, labels, rotation=17, ha='center', fontsize=10)
    plt.ylabel('Confidence')
    plt.title(f"{file_name} (α={alpha}, $\\tau_{{sim}}$ = {sim_threshold})")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    original_patch = mpatches.Patch(color='blue', label='Original Confidence')
    boosted_up_patch = mpatches.Patch(color='green', label='Boosted ↑')
    boosted_down_patch = mpatches.Patch(color='red', label='Boosted ↓/unchanged')

    plt.legend(handles=[original_patch, boosted_up_patch, boosted_down_patch], fontsize=10)
    plt.tight_layout()
    plt.show()

# === Main Clipwise Analysis ===
if __name__ == "__main__":
    target_files = [
        "5265004457.wav",
    ]
    
    for file_name in target_files:
        file_id = file_name.replace(".wav", "")
        if file_id not in captions_data or file_name not in predictions_df.index:
            print(f"Missing data for {file_name}")
            continue

        audio_captions = captions_data[file_id].get("audio_captions", [])
        caption_phrases = extract_caption_phrases(audio_captions)
        print(f"audio_captions: {audio_captions}")
        print(f"caption_phrases: {caption_phrases}")

        row = predictions_df.loc[file_name]
        audio_tags = []
        for i in range(1, 11):  # tag1 to tag10
            tag = row.get(f'tag{i}')
            prob = row.get(f'tag{i}prob')
            if isinstance(tag, str) and pd.notna(prob):
                audio_tags.append((tag, float(prob)))

        boosted_results = boost_confidence_ratio(audio_tags, caption_phrases, audio_captions, alpha=boost_alpha, caption_similarity_threshold=semantic_similarity_threshold)

        print(f"\n{'=' * 100}")
        print(f"File: {file_name}")
        print(f"Alpha = {boost_alpha}, Semantic Sim Thresh = {semantic_similarity_threshold}")
        print(f"{'-' * 100}")
        print(f"{'Tag':<30} {'Original':<10} {'Boosted':<10} {'Matches'}")
        print(f"{'-' * 100}")
        for tag, info in sorted(boosted_results.items(), key=lambda x: x[1]['boosted'], reverse=True):
            print(f"{tag:<30} {info['original']:<10.3f} {info['boosted']:<10.3f} {info['matches']}")

        ground_truth = ground_truth_tags_dict.get(file_name, [])
        plot_boosting_results(
            boosted_results,
            file_name,
            boost_alpha,
            caption_phrases,
            ground_truth,
            semantic_similarity_threshold
        )
