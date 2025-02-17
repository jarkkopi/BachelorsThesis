import json
import string

import numpy as np
import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet

import spacy

nlp = spacy.load('en_core_web_lg')

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


from IPython import embed

def cosine_similarity(embeddings, word):
    values = []
    for emb_vect in embeddings:
        values.append(np.dot(emb_vect,word)/(np.linalg.norm(emb_vect)*np.linalg.norm(word)))
    return values


def tokenize_caption(caption):
    # Prepare caption text. Remove punctuation and lower case it
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    tokenize = nlp(caption.lower())
    # Lemmatize the caption tags
    tokenize_caption = []
    for token in tokenize:
        # We do not care about the stop words 
        if not token.is_stop:
            tokenize_caption.append(token.text)

    return tokenize_caption

def get_word_synonyms(word):

    keep = []
    synsets = wordnet.synsets(word, lang='eng')
    for synset in synsets:
        word = synset.name().split(".")[0]
        if word not in keep:
            keep.append(word)
    return keep


def preprocess_entry(entry_ontology):
    entry_ontology = entry_ontology.translate(str.maketrans('', '', string.punctuation))
    tokenize = nlp(entry_ontology.lower())
    return tokenize

def collect_ontology():

    with open("audioset_ontology.json") as f:
        ontology = json.load(f)

    original_list_ontology = []
    for entry in ontology:
        if 'abstract' in entry['restrictions']:
            pass
        elif 'blacklist' in entry['restrictions']:
            pass
        else:
            entry_token = preprocess_entry(entry['name'])
            new_list = [str(t) for t in entry_token]
            original_list_ontology.append(' '.join(new_list))

    return original_list_ontology



def extract_labels(caption, embeddings, original_list_ontology):
    labels = []
    tok_caption = tokenize_caption(caption)
    for token in tok_caption:
        

        token_embeddings = tokenizer.tokenize(token)
        result = cosine_similarity(embeddings,token_embeddings)
        index_sort = np.argsort(result)
        
        if result[index_sort[-1]] >= 0.9:
            labels.append(np.array(original_list_ontology)[index_sort[-1]])

        # Find synonyms:
        else:
            synonyms = get_word_synonyms(token)
            for synonym in synonyms:
                token_embeddings = tokenizer.tokenize(synonym)
                result = cosine_similarity(embeddings,token_embeddings)
                index_sort = np.argsort(result)
                if result[index_sort[-1]] >= 0.9:
                    labels.append(np.array(original_list_ontology)[index_sort[-1]])

    return labels

def main():


    AS_ontology = collect_ontology()

    AS_embeddings = tokenizer.tokenize(AS_ontology)

    caption = 'One man is laughing in the background.'

    tags = extract_labels(caption, AS_embeddings, AS_ontology)
      
    print(f'Extracted tags: {tags}')


if __name__ == '__main__':
    main()