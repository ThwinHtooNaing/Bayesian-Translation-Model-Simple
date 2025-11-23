import collections
import math
import json
import csv
import re
from pythainlp.tokenize import word_tokenize

# ==========================================
# Helper: Language Model
# ==========================================
class SimpleLanguageModel:
    def __init__(self):
        self.bigrams = collections.defaultdict(lambda: collections.defaultdict(int))
        self.unigrams = collections.defaultdict(int)
        self.total_words = 0

    def train(self, filename):
        print(f"Training Language Model from {filename}...")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("en_text") or row.get("english")
                    if not text: continue
                    
                    # Basic cleaning
                    text = text.lower()
                    text = re.sub(r"[^a-z0-9\s]", "", text)
                    tokens = [t for t in text.split() if t.strip()]
                    
                    # Add Start/End markers
                    tokens = ["<START>"] + tokens + ["<END>"]

                    for i in range(len(tokens) - 1):
                        w1, w2 = tokens[i], tokens[i+1]
                        self.bigrams[w1][w2] += 1
                        self.unigrams[w1] += 1
                        self.total_words += 1
                        
            print(f"LM Trained. Vocab size: {len(self.unigrams)}")
        except FileNotFoundError:
            print("Error: CSV file not found. LM will be empty.")

# ==========================================
# Core: Hypothesis Class
# ==========================================
class Hypothesis:
    def __init__(self, tokens, score, covered_indices):
        self.tokens = tokens
        self.score = score
        # covered_indices must be a set for O(1) lookup, converted to tuple for hashing
        self.covered_indices = covered_indices  

    def __repr__(self):
        last = self.tokens[-1] if self.tokens else ""
        return f"Hyp('{last}', score={self.score:.2f}, cov={len(self.covered_indices)})"

# ==========================================
# Core: Multi-Stack Decoder (The Fix)
# ==========================================
class StackDecoder:
    def __init__(self, model_weights, lm_bigrams, lm_unigrams, beam_width=20, top_k_tm=5):
        self.lm_bigrams = lm_bigrams
        self.lm_unigrams = lm_unigrams
        self.beam_width = beam_width
        self.top_k_tm = top_k_tm
        self.vocab_size = max(1, len(self.lm_unigrams))

        # Load Translation Model
        self.tm_map = collections.defaultdict(list)
        
        # Input is {thai: {eng: prob}}
        count = 0
        for thai_word, eng_map in model_weights.items():
            candidates = []
            for eng_word, prob in eng_map.items():
                if prob > 0.00001: 
                    candidates.append((eng_word, prob))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            self.tm_map[thai_word] = candidates[:self.top_k_tm]
            if candidates: count += 1
            
        print(f"Decoder loaded weights for {count} Thai words.")
        self._eps = 1e-12

    def _clean_thai_tokens(self, thai_text_or_list):
        if isinstance(thai_text_or_list, list):
            toks = thai_text_or_list
        else:
            toks = word_tokenize(thai_text_or_list, engine="newmm")
        
        out = []
        for t in toks:
            t = t.replace("\u200b", "").replace("\ufeff", "").strip()
            if t != "":
                out.append(t)
        return out

    def _lm_log_prob(self, w1, w2):
        # Laplace smoothing
        bigram_count = self.lm_bigrams.get(w1, {}).get(w2, 0)
        unigram_count = self.lm_unigrams.get(w1, 0)
        denom = unigram_count + self.vocab_size + 1
        prob = (bigram_count + 1) / denom
        return math.log(prob)

    def decode(self, thai_text):
        thai_tokens = self._clean_thai_tokens(thai_text)
        n = len(thai_tokens)
        
        # ============================================================
        # MULTI-STACK ALGORITHM
        # stacks[k] contains hypotheses that have covered exactly k source words.
        # This ensures we compare apples to apples (same length coverage).
        # ============================================================
        
        # Each stack is a Dict: Key=(last_eng_word, covered_tuple) -> Value=Hypothesis
        # This performs automatic recombination (keeping best path to same state)
        stacks = [{} for _ in range(n + 1)]

        # Initialize Stack 0
        start_hyp = Hypothesis(["<START>"], 0.0, set())
        stacks[0][("<START>", tuple())] = start_hyp

        # Iterate through stacks 0 to N-1
        for i in range(n):
            # 1. Get hypotheses from current stack
            if not stacks[i]:
                continue
                
            # 2. Pruning: Keep top K hypotheses in this stack
            current_hyps = sorted(stacks[i].values(), key=lambda h: h.score, reverse=True)
            current_hyps = current_hyps[:self.beam_width]

            # 3. Expand each hypothesis
            for hyp in current_hyps:
                prev_eng_word = hyp.tokens[-1]

                # Try to translate every UNCOVERED Thai word
                for thai_idx in range(n):
                    if thai_idx in hyp.covered_indices:
                        continue

                    thai_word = thai_tokens[thai_idx]
                    
                    # Get Translation Candidates
                    candidates = self.tm_map.get(thai_word, [])
                    
                    # Handle OOV (Out Of Vocabulary)
                    if not candidates:
                        # Pass-through with penalty
                        candidates = [(thai_word, 0.0001)]

                    for eng_word, tm_prob in candidates:
                        # --- Score Calculation ---
                        # A. Language Model (P(eng | prev))
                        lm_score = self._lm_log_prob(prev_eng_word, eng_word)
                        
                        # B. Translation Model (P(thai | eng) * P(align) implicitly)
                        # Note: math.log(prob) will be negative
                        tm_score = math.log(tm_prob)
                        
                        new_score = hyp.score + lm_score + tm_score
                        
                        # Create New Hypothesis
                        new_covered_set = hyp.covered_indices | {thai_idx}
                        new_hyp = Hypothesis(
                            hyp.tokens + [eng_word], 
                            new_score, 
                            new_covered_set
                        )
                        
                        # --- Add to Next Stack (i + 1) ---
                        # Key for Recombination: (last_word, covered_mask)
                        # If we reached same state with better score, keep better one.
                        next_stack_idx = len(new_covered_set)
                        state_key = (eng_word, tuple(sorted(new_covered_set)))
                        
                        existing = stacks[next_stack_idx].get(state_key)
                        if existing is None or new_score > existing.score:
                            stacks[next_stack_idx][state_key] = new_hyp

        # ============================================================
        # Finalize: Look at Stack[N] (All words covered)
        # ============================================================
        final_hyps = []
        
        # If Stack[N] is empty (maybe due to overly aggressive pruning?), 
        # fallback to the furthest non-empty stack.
        target_stack = stacks[n]
        if not target_stack:
            for i in range(n-1, -1, -1):
                if stacks[i]:
                    target_stack = stacks[i]
                    break
        
        # Add <END> token to valid hypotheses
        for hyp in target_stack.values():
            lm_score = self._lm_log_prob(hyp.tokens[-1], "<END>")
            final_hyp = Hypothesis(
                hyp.tokens + ["<END>"],
                hyp.score + lm_score,
                hyp.covered_indices
            )
            final_hyps.append(final_hyp)

        if not final_hyps:
            return [], 0.0

        # Pick Best
        final_hyps.sort(key=lambda h: h.score, reverse=True)
        best_hyp = final_hyps[0]
        
        # Strip markers
        result_tokens = [t for t in best_hyp.tokens if t not in ("<START>", "<END>")]
        
        return result_tokens, best_hyp.score

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    print("Loading Translation Model...")
    try:
        with open("model_weights_th_en.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            tm_weights = data.get("translation", {})
    except FileNotFoundError:
        print("Error: model_weights_th_en.json not found. Run trainer.py first.")
        tm_weights = {}

    lm = SimpleLanguageModel()
    lm.train("nus_sms.csv")

    decoder = StackDecoder(
        model_weights=tm_weights,
        lm_bigrams=lm.bigrams,
        lm_unigrams=lm.unigrams,
        beam_width=20,  
        top_k_tm=5      
    )

    print("\n--- Corrected Stack Decoder Results ---")
    test_sentences = [
        "สวัสดีครับ",       # Hello
        "ขอบคุณมาก",        # Thank you very much
        "ฉันรักคุณ",        # I love you
        "รอสักครู่",        # Wait a moment
        "ไม่เข้าใจ"         # Don't understand
    ]

    for sent in test_sentences:
        decoded_tokens, score = decoder.decode(sent)
        result_text = " ".join(decoded_tokens)
        print(f"Thai: {sent}")
        print(f"Eng : {result_text} (Score: {score:.2f})")
        print("-" * 30)