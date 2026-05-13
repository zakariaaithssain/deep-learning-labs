import spacy
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

class SentimentDataPreprocessor:
    def __init__(self, max_words=5000, maxlen=None, oov_token='<OOV>'):
        self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        self.max_words = max_words
        self.maxlen = maxlen
        self.tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
        
        #preserve negations
        negations = ['not', 'no', 'never', 'neither', 'nor']
        for word in negations:
            self.nlp.Defaults.stop_words.discard(word)
            self.nlp.vocab[word].is_stop = False
    
    def preprocess_text(self, text):
        text = text.lower()
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc 
                  if not token.is_stop 
                  and not token.is_punct 
                  and token.is_alpha]
        return ' '.join(tokens)
    
    def fit(self, texts):
        #preprocess all texts
        preprocessed = [self.preprocess_text(text) for text in texts]
        
        #fit tokenizer
        self.tokenizer.fit_on_texts(preprocessed)
        
        return self
    
    def transform(self, texts, padding='post'):
        """Transform texts to padded sequences"""
        # Preprocess
        preprocessed = [self.preprocess_text(text) for text in texts]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(preprocessed)
        
        # Pad
        X = pad_sequences(sequences, maxlen=self.maxlen, padding=padding)
        
        return X
    
    def fit_transform(self, texts, padding='post'):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts, padding)

# Usage
preprocessor = SentimentDataPreprocessor(max_words=5000)

# Training data
train_reviews = [
    "This product is great!",
    "Not good at all",
    "Amazing quality",
    "Terrible experience",
    "Excellent, highly recommend"
]
train_labels = [1, 0, 1, 0, 1]

# Fit and transform
X_train = preprocessor.fit_transform(train_reviews)
y_train = np.array(train_labels)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("\nPadded sequences:")
print(X_train)

# New test data
test_reviews = [
    "Outstanding product!",
    "Not satisfied"
]

X_test = preprocessor.transform(test_reviews)
print(f"\nX_test shape: {X_test.shape}")
print(X_test)