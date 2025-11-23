import collections
import math
import csv
import pickle
import os
from pythainlp.tokenize import word_tokenize

# Try imports to handle cases where dependencies might be missing during initial setup
try:
    from decoder import StackDecoder   # Make sure StackDecoder.py is in the same folder
    from trainer import EMTrainer      # Make sure EMTrainer.py is in the same folder
except ImportError:
    StackDecoder = None
    EMTrainer = None
    print("Warning: 'decoder' or 'trainer' modules not found. Ensure StackDecoder.py and EMTrainer.py are present.")

class BayesianTranslator:
    def __init__(self):
        # Language Model counts (P(English))
        self.lm_bigrams = collections.defaultdict(int)
        self.lm_unigrams = collections.defaultdict(int)
        self.total_words = 0

        # Translation table (P(Thai|English))
        self.translation_table = {}

    # --------------------------------------------------------------
    #  HELPER: TOKENIZE THAI
    # --------------------------------------------------------------
    def tokenize_thai(self, text):
        if not text:
            return []
        # 'newmm' is the standard dictionary-based tokenizer for Thai
        tokens = word_tokenize(text, engine="newmm")
        # Remove zero-width spaces or BOM if present
        return [t.strip() for t in tokens if t.strip() and t not in ["\u200b", "\ufeff"]]

    # --------------------------------------------------------------
    #  HELPER: CLEAN ENGLISH
    # --------------------------------------------------------------
    def clean_english(self, text):
        text = text.lower()
        return text.split()

    # --------------------------------------------------------------
    #  TRAIN LANGUAGE MODEL (P(English))
    # --------------------------------------------------------------
    def train_lm_from_csv(self, csv_file):
        print(f"--- Training Language Model from {csv_file} ---")
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    raw_eng = row.get("en_text", "")
                    if not raw_eng:
                        continue
                    
                    # Add boundary markers
                    tokens = ["<START>"] + self.clean_english(raw_eng) + ["<END>"]
                    
                    for i in range(len(tokens) - 1):
                        self.lm_bigrams[(tokens[i], tokens[i+1])] += 1
                        self.lm_unigrams[tokens[i]] += 1
                    
                    self.lm_unigrams[tokens[-1]] += 1
                    self.total_words += len(tokens)
                    count += 1
            print(f"Language Model trained with {count} sentences.")
        except Exception as e:
            print(f"[Error] LM Training failed: {e}")

    # --------------------------------------------------------------
    #  SAVE / LOAD MODELS (FIXED FOR PICKLE ERROR)
    # --------------------------------------------------------------
    def save_model(self, filename="translator_model.pkl"):
        print(f"Saving model to {filename}...")
        
        # Helper to recursively convert defaultdict to dict to remove lambdas
        def recursive_to_dict(obj):
            if isinstance(obj, dict): # This covers dict and defaultdict
                return {k: recursive_to_dict(v) for k, v in obj.items()}
            return obj

        # CRITICAL: Convert all defaultdicts (which have lambdas) to standard dicts
        clean_translation_table = recursive_to_dict(self.translation_table)
        clean_lm_bigrams = dict(self.lm_bigrams)
        clean_lm_unigrams = dict(self.lm_unigrams)

        data = {
            "lm_bigrams": clean_lm_bigrams,
            "lm_unigrams": clean_lm_unigrams,
            "total_words": self.total_words,
            "translation_table": clean_translation_table
        }
        
        try:
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename="translator_model.pkl"):
        print(f"Loading model from {filename}...")
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            
            # Restore Language Models as defaultdict(int) so logic elsewhere (+=1) works if needed
            self.lm_bigrams = collections.defaultdict(int, data["lm_bigrams"])
            self.lm_unigrams = collections.defaultdict(int, data["lm_unigrams"])
            self.total_words = data["total_words"]
            
            # Translation table acts as a read-only dict during translation, so dict is fine
            self.translation_table = data["translation_table"]
            
            print("Model loaded successfully.")
            return True
        except FileNotFoundError:
            print("Model file not found.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # --------------------------------------------------------------
    #  TRANSLATE USING STACK DECODER
    # --------------------------------------------------------------
    def translate(self, thai_sentence):
        if not self.translation_table:
            print("[Error] Translation table not loaded.")
            return "", 0.0

        if not StackDecoder:
            print("[Error] StackDecoder module is missing.")
            return "", 0.0

        # STEP 1: Tokenize the input Thai sentence
        thai_tokens = self.tokenize_thai(thai_sentence)
        
        if not thai_tokens:
            return "", 0.0

        # STEP 2: Initialize Decoder
        decoder = StackDecoder(
            model_weights=self.translation_table,
            lm_bigrams=self.lm_bigrams,
            lm_unigrams=self.lm_unigrams,
            beam_width=10,
            top_k_tm=5
        )

        # STEP 3: Decode
        try:
            tokens, score = decoder.decode(thai_tokens)
            return " ".join(tokens), score
        except Exception as e:
            print(f"Decoding error: {e}")
            return "", 0.0

if __name__ == "__main__":
    translator = BayesianTranslator()
    csv_filename = "nus_sms.csv"
    model_filename = "translator_model.pkl"

    # Check if a trained model already exists
    if os.path.exists(model_filename):
        print("Found saved model. Loading...")
        translator.load_model(model_filename)
    else:
        print("No saved model found. Starting training...")
        
        if EMTrainer and os.path.exists(csv_filename):
            # 1. Train EM Model
            em_trainer = EMTrainer()
            em_trainer.load_data(csv_filename)
            em_trainer.initialize_uniform()
            
            print("Training IBM Model 1 (10 iterations)...")
            em_trainer.train_model1(iterations=10)
            
            print("Training IBM Model 2 (5 iterations)...")
            em_trainer.train_model2(iterations=5) 
            
            # Transfer probabilities to translator
            translator.translation_table = em_trainer.t

            # 2. Train Language Model
            translator.train_lm_from_csv(csv_filename)
            
            # 3. Save for next time
            translator.save_model(model_filename)
        else:
            print(f"Error: Cannot train. Ensure '{csv_filename}' exists and EMTrainer is imported.")
            exit()

    # ------------------------
    # 3. Test / Interactive
    # ------------------------
    test_sentences = [
        "เจอกันใหม่",
        "ขอบคุณ",
        "ไปหาที่ทำอะไรสนุกๆหลังจากนั้นไหม"
    ]

    print("\n--- BATCH TEST RESULTS ---")
    for t in test_sentences:
        eng, score = translator.translate(t)
        print(f"Thai: {t}")
        print(f"Pred: {eng} | Score: {score:.4f}")
        print("-" * 30)

    # Optional: Interactive loop
    while True:
        user_input = input("\nEnter Thai sentence (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        eng, score = translator.translate(user_input)
        print(f"Translation: {eng} (Score: {score:.4f})")