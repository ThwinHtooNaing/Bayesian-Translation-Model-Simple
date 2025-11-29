import collections
import csv
import json
from pythainlp.tokenize import word_tokenize
import re

class ThaiToEngTrainer:
    def __init__(self):
        # P(eng|thai) - Lexical probabilities: self.t[thai_word][eng_word]
        self.t = collections.defaultdict(lambda: collections.defaultdict(float)) 
        # P(j|i,l,m) - Alignment probabilities
        self.a = collections.defaultdict(lambda: collections.defaultdict(float)) 
        self.sentences = [] 
    
    def clean_english(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = text.split()
        return [t for t in tokens if t.strip() != ""]
    
    def clean_thai_tokens(self, tokens):
        clean = []
        for t in tokens:
            t = t.replace("\u200b", "")
            t = t.replace("\ufeff", "")
            t = t.strip()
            if t != "":
                clean.append(t)
        return clean

    def load_data(self, filename):
        print(f"Loading data from {filename}...")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    thai_raw = row.get("th_text") or row.get("thai")
                    eng_raw = row.get("en_text") or row.get("english")

                    if not thai_raw or not eng_raw:
                        continue

                    # 1. Tokenize Thai (Source)
                    t_tok = word_tokenize(thai_raw, engine="newmm")
                    t_tok = self.clean_thai_tokens(t_tok)
                    
                    # IMPORTANT: In IBM Models, the SOURCE language (Thai) gets the NULL token
                    t_tok = ["NULL"] + t_tok 

                    # 2. Tokenize English (Target)
                    e_tok = self.clean_english(eng_raw)
                    # English does NOT get NULL in this direction

                    if t_tok and e_tok:
                        self.sentences.append((t_tok, e_tok))

            print(f"Loaded {len(self.sentences)} sentences.")
        
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")

    def initialize_uniform(self):
        print("Initializing uniform probabilities (Thai -> Eng)...")
        # self.t[thai_word][eng_word]
        for t_sent, e_sent in self.sentences:
            for t_word in t_sent:
                for e_word in e_sent:
                    self.t[t_word][e_word] = 1.0

        # Normalize so sum(P(e|t)) = 1 for each t
        for t_word, e_map in self.t.items():
            total = len(e_map)
            for e_word in e_map:
                self.t[t_word][e_word] = 1.0 / total

    def train_model1(self, iterations=10):
        print(f"\n--- Training IBM Model 1 ({iterations} iterations) ---")

        for it in range(iterations):
            count = collections.defaultdict(float)
            total_t = collections.defaultdict(float) # Total counts for Source (Thai)

            for t_sent, e_sent in self.sentences:
                
                
                for e_word in e_sent:
                    # Calculate Normalization Factor Z (Sum over Source/Thai)
                    Z = sum(self.t[t_word][e_word] for t_word in t_sent)
                    
                    if Z == 0: continue

                    for t_word in t_sent:
                        delta = self.t[t_word][e_word] / Z
                        count[(t_word, e_word)] += delta
                        total_t[t_word] += delta

            for (t_word, e_word), val in count.items():
                self.t[t_word][e_word] = val / total_t[t_word]

            print(f"Model 1 Iteration {it+1} complete.")


    def train_model2(self, iterations=10):
        print(f"\n--- Training IBM Model 2 ({iterations} iterations) ---")

        for it in range(iterations):
            count_t = collections.defaultdict(float) # Count lexical
            total_t = collections.defaultdict(float) # Total lexical (Source)

            count_a = collections.defaultdict(float) # Count alignment
            total_a = collections.defaultdict(float) # Total alignment

            for t_sent, e_sent in self.sentences:
                l = len(t_sent) # Source Length (Thai)
                m = len(e_sent) # Target Length (English)

                # i = position in Target (English)
                # j = position in Source (Thai)

                for i, e_word in enumerate(e_sent):
                    Z = 0.0
                    # E-Step Part 1: Calculate Z
                    for j, t_word in enumerate(t_sent):
                        # alignment prob: a(j | i, l, m) -> probability that English pos i aligns to Thai pos j
                        align_prob = self.a[(i, l, m)].get(j, 1.0 / (l + 1))
                        Z += self.t[t_word][e_word] * align_prob
                    
                    if Z == 0: continue

                    # E-Step Part 2: Collect Counts
                    for j, t_word in enumerate(t_sent):
                        align_prob = self.a[(i, l, m)].get(j, 1.0 / (l + 1))
                        delta = (self.t[t_word][e_word] * align_prob) / Z

                        # Lexical counts
                        count_t[(t_word, e_word)] += delta
                        total_t[t_word] += delta

                        # Alignment counts
                        count_a[(j, i, l, m)] += delta
                        total_a[(i, l, m)] += delta

            for (t_word, e_word), val in count_t.items():
                self.t[t_word][e_word] = val / total_t[t_word]

            for (j, i, l, m), val in count_a.items():
                self.a[(i, l, m)][j] = val / total_a[(i, l, m)]

            print(f"Model 2 Iteration {it+1} complete.")

    def get_best_translation(self, thai_word):
        if thai_word not in self.t:
            return "UNKNOWN"
        candidates = sorted(self.t[thai_word].items(), key=lambda x: x[1], reverse=True)
        return candidates[0] 

    def save_model(self, filename="model_weights_th_en.json"):
        cleaned_t = {}
        # Structure: { "thai_word": { "eng_word": 0.5, ... } }
        for t_word, e_map in self.t.items():
            filtered = {k: v for k, v in e_map.items() if v > 0.001}
            if filtered:
                cleaned_t[t_word] = filtered

        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"translation": cleaned_t}, f, ensure_ascii=False, indent=2)
        print(f"Model saved to {filename}")

if __name__ == "__main__":
    trainer = ThaiToEngTrainer()
    
    trainer.load_data("nus_sms.csv")
    
    trainer.initialize_uniform()

    trainer.train_model1(iterations=10)
    trainer.train_model2(iterations=5)

    trainer.save_model("model_weights_th_en.json")

    print("\n--- Test Translations (Thai -> Eng) ---")
    # Use Thai words here to test
    test_words = ["สวัสดี", "ขอบคุณ", "พรุ่งนี้", "รอ", "รัก"]
    
    for w in test_words:
        print(f"{w} → {trainer.get_best_translation(w)}")