CS5720 Neural Networks and Deep Learning - Spring 2025
Home Assignment 4
Student Name: Satwika Kallem

Assignment Description
This assignment covers concepts from Chapters 9 and 10, focusing on Natural Language Processing (NLP) and Transformer models. It includes implementing basic NLP preprocessing, Named Entity Recognition (NER) with spaCy, the scaled dot-product attention mechanism, and sentiment analysis using Hugging Face Transformers.

Code Explanation
Q1: NLP Preprocessing Pipeline

Description:

The preprocess_nlp function takes a sentence as input and performs the following preprocessing steps:

Tokenization: Splits the sentence into individual words and punctuation using word_tokenize.

Stopword Removal: Removes common English stopwords (e.g., "the", "in", "are") using the stopwords corpus from NLTK.

Stemming: Reduces words to their root form using the Porter Stemmer from NLTK.

The main part of the script downloads the necessary NLTK data (stopwords and tokenizer) and then calls the function with the provided sentence.

Code:

See the "NLP Preprocessing Pipeline" section in the main document.

Key Points:

Uses NLTK for tokenization, stopword removal, and stemming.

Prints the original tokens, tokens without stopwords, and the final stemmed words.

Q2: Named Entity Recognition with SpaCy

Description:

The extract_named_entities function uses the spaCy library to identify and extract named entities from a given sentence.

For each entity, it prints the entity text, label (e.g., PERSON, DATE, GPE), start character position, and end character position.

The main part of the script downloads the spaCy English language model (if not already downloaded) and processes the input sentence.

Code:

See the "Named Entity Recognition with SpaCy" section in the main document.

Key Points:

Uses spaCy's pre-trained English language model (en_core_web_sm).

Iterates through the detected entities and prints their attributes.

Q3: Scaled Dot-Product Attention

Description:

The scaled_dot_product_attention function implements the scaled dot-product attention mechanism, a core component of Transformer models.

It takes Query (Q), Key (K), and Value (V) matrices as input and performs the following steps:

Computes the dot product of Q and K transposed.

Scales the result by the square root of the key dimension (d_k).

Applies the softmax function to obtain attention weights.

Computes the weighted sum of the Value (V) matrix using the attention weights.

The main part of the script defines example Q, K, and V matrices and calls the function.

Code:

See the "Scaled Dot-Product Attention" section in the main document.

Key Points:

Uses NumPy for matrix operations.

Returns the attention weights and the output matrix.

Q4: Sentiment Analysis using Hugging Face Transformers

Description:

The analyze_sentiment function uses the Hugging Face transformers library to perform sentiment analysis on a given sentence.

It loads a pre-trained sentiment analysis pipeline and uses it to classify the input sentence as either positive or negative.

The main part of the script defines the input sentence and calls the function and prints the label and confidence score.

Code:

See the "Sentiment Analysis using HuggingFace Transformers" section in the main document.

Key Points:

Uses the pipeline function from Hugging Face transformers to load a pre-trained model.

Prints the sentiment label (e.g., "POSITIVE", "NEGATIVE") and the confidence score, rounded to four decimal places.

Short Answer Questions
Q1: NLP Preprocessing Pipeline

What is the difference between stemming and lemmatization? Provide examples with the word “running.”

Stemming: A simpler process that removes suffixes from words to reduce them to a common base form (stem). The stem may not be a valid word.

Example: running -> run

Lemmatization: A more sophisticated process that reduces words to their base or dictionary form (lemma). Lemmatization considers the word's meaning and context and the lemma is always a valid word.

Example: running -> run

Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?
* Useful: Removing stop words can reduce noise and focus on the most important words in a text, which can improve the performance of tasks like text classification, information retrieval.
* Harmful: Removing stop words can remove important meaning from the sentence.  For example, in sentiment analysis, removing "not" can flip the meaning of a sentence.  In question answering, removing "who", "what", "where" would make the question impossible to understand.

Q2: Named Entity Recognition with SpaCy

How does NER differ from POS tagging in NLP?

NER (Named Entity Recognition): Identifies and classifies named entities in a text into predefined categories, such as person, organization, date, or location.

POS (Part-of-Speech) Tagging: Labels each word in a text with its corresponding part of speech, such as noun, verb, adjective, or adverb.

Describe two applications that use NER in the real world (e.g., financial news, search engines).

Financial News: NER can identify companies, stock tickers, and monetary values in news articles, which can be used to analyze market trends, detect fraud, and automate trading.

Search Engines: NER can help search engines understand the intent behind a search query. For example, if you search for "Barack Obama", the search engine can use NER to identify "Barack Obama" as a person and provide more relevant results.

Q3: Scaled Dot-Product Attention

Why do we divide the attention score by √d in the scaled dot-product attention formula?

To prevent the dot products from becoming too large as the dimensionality (d) of the keys and queries increases. Large dot products can lead to extremely small or large values after the softmax function, making gradients vanish during training. Scaling by the square root of d helps to stabilize the gradients and improve training.

How does self-attention help the model understand relationships between words in a sentence?

Self-attention allows a model to weigh the importance of different words in a sentence when processing a specific word. By calculating attention weights, the model can determine which words are most relevant to the current word, enabling it to capture long-range dependencies and understand the context and relationships between words.

Q4: Sentiment Analysis using Hugging Face Transformers

What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?

BERT (Bidirectional Encoder Representations from Transformers): Uses a transformer encoder architecture.

GPT (Generative Pre-trained Transformer): Uses a transformer decoder architecture.

Explain why using pre-trained models (like BERT or GPT) is beneficial for NLP applications instead of training from scratch.

Pre-trained models are trained on massive amounts of text data, allowing them to learn rich and general-purpose language representations.  Using these models as a starting point for NLP applications offers several advantages:

Reduced data requirements: Pre-trained models require less task-specific training data.

Improved performance: They often achieve higher accuracy and better generalization than models trained from scratch.

Faster development: They save significant time and computational resources compared to training a large model from scratch.

