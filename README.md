# Aspect Based Sentiment Analysis (ABSA) for Customer Reviews

This repository contains the implementation and comparative study of various Natural Language Processing (NLP) models designed to automate the analysis of customer feedback. Rather than assigning a single sentiment to an entire review, this project utilizes Aspect Based Sentiment Analysis (ABSA) to categorize reviews into specific Entity-Aspect pairs (e.g., `Food#Quality`, `Service#General`) and determine the sentiment for each individual aspect. 

## Core Project Architecture

The pipeline processes complex sentences containing multiple distinct opinions using several core components:

*   **Constituency Parsing:** A single sentence can contain multiple aspects with different sentiments. We utilized the Stanford CoreNLP Stanza library for constituency parsing, which breaks complex sentences down into smaller sub-sentences (constituents) belonging to specific grammar categories. This allows the model to isolate and evaluate multiple aspects independently.
*   **Word Embeddings:** We utilized the Continuous Bag of Words (CBOW) method (a Word2Vec technique) to capture contextual semantic relationships. Ultimately, pre-trained Google word embeddings with 300 dimensions were implemented, as our custom word vectors did not provide sufficient accuracy.
*   **Speech-to-Text (STT) Conversion:** As a complement to the ABSA model, we integrated a speech-to-text feature to process spoken reviews. The Python `SpeechRecognition` library was chosen over MATLAB alternatives to avoid the subscription fees associated with third-party APIs like Google Speech API and Microsoft Azure.

## Models Evaluated & Results

Throughout the project, multiple models were trained and tested to find the optimal solution for Aspect Detection and Sentiment Classification. 

### 1. Aspect Category Detection
*   **Blank spaCy Model:** Evaluated for Custom Named Entity Recognition (NER). 
    *   *Result:* Discarded due to a low accuracy of 77%. The model failed to recognize specific food items (like "Chow Fun") and demonstrated a heavy bias toward common western food nouns.
*   **Bi-directional LSTM:** Evaluated to process sequential data in both forward and backward contexts. 
    *   *Result:* Discarded despite a high theoretical accuracy of 98%. The model suffered from a severe class imbalance, causing it to overpredict the 'O' (NULL) tag for almost all aspects.
*   **Pre-trained BERT Model (Base):** Utilizes an attention mechanism to look at word context from both sides simultaneously. 
    *   *Result:* **Selected for the final model.** It achieved high accuracy and successfully pinpointed all aspects after resolving initial class imbalance issues.

### 2. Sentiment Polarity Classification
*   **Logistic Regression:** Evaluated using TF-IDF vectorization on the SemEval2016 dataset. 
    *   *Result:* Discarded due to poor accuracy (77%). The model struggled with negations and blindly classified phrases like "not bad" as negative because of the individual word weights.
*   **Bi-directional LSTM:** Evaluated using the IMDB reviews dataset. The architecture included multiple layers ending in a dense layer with a sigmoid activation function. 
    *   *Result:* **Selected for the final model.** It achieved a high accuracy of 92.02% and correctly classified contextual sentiments.

## Development Challenges

Building the pipeline involved navigating several significant challenges:

*   **Dataset Limitations:** The datasets suffered from severe class imbalances (overloading the 'O' tag), lacked proper BIO/IOB tagging for spaCy, and restricted sentiment inputs to binary labels.
*   **Preprocessing Pitfalls:** Blindly removing stop words initially stripped out vital negations (like "not" or "never"), completely altering sentence meanings and ruining accuracy.
*   **Hardware Constraints:** Fine-tuning the BERT model placed a heavy load on the machine due to its complexity and long training times, restricting the ability to run more epochs or select higher parameters.
*   **Model Dependencies:** Saving the BERT model in Windows caused a multi-threading infinite loop error, requiring a switch to Ubuntu to resolve.

## Conclusion

This project successfully proposes a complete pipeline for Aspect Based Sentiment Analysis. Based on our comparative testing, the final combined model pairs the **BERT Model** for robust Aspect Category Detection with the **Bi-directional LSTM Model** for Sentiment Polarity Classification, supplemented by Stanford CoreNLP Stanza for parsing and Python `SpeechRecognition` for audio input.
