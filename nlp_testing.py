import json
import spacy

json_file_path = './data/test_captions.json'
nlp = spacy.load('en_core_web_sm')

with open(json_file_path, 'r') as f:
    captions_data = json.load(f)

def extract_nouns_and_verbs(captions_data):
    connected_words = {}

    for key, captions in list(captions_data.items())[:10]:  # first ten
        all_captions = captions.get("audio_captions", []) + captions.get("visual_captions", []) + captions.get("audio_visual_captions", []) + captions.get("GPT_AV_captions", [])
        connected_words[key] = {"nouns": [], "verbs": []}

        for caption in all_captions:
            doc = nlp(caption)
            """
            for token in doc:
                if token.pos_ in {"NOUN", "VERB"} and token.head.pos_ in {"NOUN", "VERB"}:
                    connected_words[key].append((token.text, token.head.text))
            """
            for token in doc:
                if token.pos_ == "NOUN":
                    connected_words[key]["nouns"].append(token.text)
                elif token.pos_ == "VERB":
                    connected_words[key]["verbs"].append(token.text)
    return connected_words

connected_words = extract_nouns_and_verbs(captions_data)

for key, words in connected_words.items():
    print(f"Key: {key}")
    """
    print(f"Connected Nouns and Verbs: {words}")
    """
    print(f"Nouns: {words['nouns']}")
    print(f"Verbs: {words['verbs']}")
