# Bachelor's thesis
Bachelor's thesis experimental part.
This bachelor's thesis was expanded on, through which a workshop paper was created for DCASE workshop 2025.

Abstract—Many audio datasets lack proper audio tags, meaning the
 annotations are not complete or accurate, limiting their usability. While
 pre-trained sound event detection or audio tagging models can predict
 tags, their accuracy may suffer due to domain mismatch and fixed label
 sets. Audio captions, which are natural language descriptions of the audio
 scene, offer rich semantic information that can reduce ambiguity and
 improve audio tagging. This study explores how captions can support
 audio tagging by increasing the confidence of the model’s predictions.
 We propose a method that uses natural language processing to compare
 phrases extracted from audio captions with tags from a pre-trained
 sound event tagging model. The comparison is done using Sentence
BERT embeddings and cosine similarity. Confidence scores are then
 adjusted based on the similarity between tags and the weighting of
 caption information. The results on the AVCaps dataset demonstrate that
 using information from captions as a postprocessing step leads to an
 improvement in F1 score, reaching 0.57, compared to a baseline model
 without caption-based postprocessing, which achieves an F1 score of 0.53.
 These findings suggest that captions can serve as a valuable complementary
 source of information to improve the reliability and robustness of audio
 tagging systems.
 Index Terms—Sound event detection, audio captioning, NLP, tagging

The used dataset can be found here: https://zenodo.org/records/14536325
