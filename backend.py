import string
import numpy as np
import tensorflow as tf
from scipy import interpolate
import pygtrie
from wordfreq import word_frequency, top_n_list

# ==========================================
# --- CONFIGURATION ---
# ==========================================
# ALPHA controls the trust ratio between the drawn stroke and the dictionary.
# 1.0 = 100% trust in the drawing (ignore dictionary completely).
# 0.0 = 100% trust in the dictionary (ignore drawing completely).
# 0.6 to 0.8 is usually the sweet spot for a smart keyboard.
ALPHA = 0.7  

MODEL_PATH = 'unistroke_hybrid_model.keras'
TARGET_LEN = 60
VOCAB_SIZE = 10000
# ==========================================
class ContextEngine:
    """Handles dictionary lookups and next-character probabilities."""
    def __init__(self, vocab_size=10000):
        self.trie = pygtrie.CharTrie()
        self.alphabet = string.ascii_lowercase
        self.vocab_size = vocab_size
        self._build_trie()

    def _build_trie(self):
        print(f"Loading top {self.vocab_size} words into Context Engine...")
        common_words = top_n_list('en', self.vocab_size)
        
        for word in common_words:
            if word.isalpha():
                self.trie[word] = word_frequency(word, 'en')
        print("Context Engine ready.")

    def get_next_char_probabilities(self, prefix):
        char_scores = {char: 0.0 for char in self.alphabet}
        prefix = prefix.lower()
        if not prefix: return {char: 1.0/26.0 for char in self.alphabet}

        try:
            matching_words = self.trie.keys(prefix=prefix)
        except KeyError:
            return {char: 1.0/26.0 for char in self.alphabet}

        total_weight = 0.0
        for word in matching_words:
            if len(word) > len(prefix):
                next_char = word[len(prefix)]
                weight = self.trie[word]
                char_scores[next_char] += weight
                total_weight += weight
                
        if total_weight > 0:
            for char in char_scores: char_scores[char] /= total_weight
        else:
            char_scores = {char: 1.0/26.0 for char in self.alphabet}
        return char_scores

    # --- NEW METHOD ADDED FOR WORD SUGGESTIONS ---
    def get_top_4_words(self, prefix):
        """Returns the top 4 most common words starting with the prefix."""
        if not prefix:
            return []
        prefix = prefix.lower()
        try:
            # Get all words matching the prefix with their frequencies
            matching_words = list(self.trie.iteritems(prefix=prefix))
            # Sort by frequency (highest first)
            matching_words.sort(key=lambda x: x[1], reverse=True)
            # Return just the top 4 word strings
            return [word for word, freq in matching_words[:4]]
        except KeyError:
            return []

class UnistrokeEngine:
    """The main backend engine that fuses stroke recognition and context."""
    def __init__(self):
        self.context_engine = ContextEngine(vocab_size=VOCAB_SIZE)
        self.int_to_char = {i: c for i, c in enumerate(string.ascii_lowercase)}
        
        print(f"Loading Neural Network from '{MODEL_PATH}'...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Neural Network loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model. {e}")
            self.model = None

    def _resample_and_extract_features(self, sequence):
        seq = np.array(sequence)
        if len(seq) < 2: return np.zeros((TARGET_LEN, 4)) 
        
        distance = np.cumsum(np.sqrt(np.sum(np.diff(seq, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0)
        if distance[-1] == 0: return np.zeros((TARGET_LEN, 4))
        distance /= distance[-1]

        interpolator = interpolate.interp1d(distance, seq, axis=0)
        resampled = interpolator(np.linspace(0, 1, TARGET_LEN))
        
        resampled -= np.mean(resampled, axis=0)
        max_val = np.max(np.abs(resampled))
        if max_val > 0: resampled /= max_val
            
        dx_dy = np.gradient(resampled, axis=0)
        return np.hstack((resampled, dx_dy))

    def get_top_4_predictions(self, raw_stroke, current_word_prefix=""):
        if not self.model or len(raw_stroke) < 2:
            return [('e', 0.0), ('t', 0.0), ('a', 0.0), ('o', 0.0)]

        raw_coords = [[pt["x"], pt["y"]] for pt in raw_stroke]
        features = self._resample_and_extract_features(raw_coords)
        features_array = np.array([features], dtype='float32') 
        
        stroke_preds = self.model.predict(features_array, verbose=0)[0]
        context_preds_dict = self.context_engine.get_next_char_probabilities(current_word_prefix)
        
        final_scores = []
        for i, char in enumerate(string.ascii_lowercase):
            stroke_prob = stroke_preds[i]
            context_prob = context_preds_dict[char]
            combined_score = (ALPHA * stroke_prob) + ((1.0 - ALPHA) * context_prob)
            final_scores.append((char, combined_score))
            
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:4]

    # --- NEW METHOD EXPOSED ---
    def get_word_suggestions(self, prefix):
        return self.context_engine.get_top_4_words(prefix)
# ==========================================
# Quick Test Block (Only runs if you execute this file directly)
# ==========================================
if __name__ == "__main__":
    print("Testing the Unistroke Backend...")
    engine = UnistrokeEngine()
    
    # Mocking a stroke that sort of looks like an 'e' or 'o'
    dummy_stroke = [{"x": np.sin(i), "y": np.cos(i)} for i in np.linspace(0, 2*np.pi, 20)]
    
    # Test 1: No prefix
    print("\nTest 1: Start of word (Prefix: '')")
    preds = engine.get_top_4_predictions(dummy_stroke, prefix="")
    for char, score in preds:
        print(f"  {char.upper()}: {score:.4f}")
        
    # Test 2: With prefix
    print("\nTest 2: Middle of word (Prefix: 'th')")
    preds = engine.get_top_4_predictions(dummy_stroke, current_word_prefix="th")
    for char, score in preds:
        print(f"  {char.upper()}: {score:.4f}")