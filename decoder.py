import collections
import math
from pythainlp.tokenize import word_tokenize

class Hypothesis:
    def __init__(self, tokens, score, covered_indices):
        self.tokens = tokens
        self.score = score
        self.covered_indices = covered_indices  # set of indices (ints)

    def __repr__(self):
        last = self.tokens[-1] if self.tokens else ""
        return f"Hyp('{last}', score={self.score:.4f}, cov={sorted(self.covered_indices)})"

class StackDecoder:
    def __init__(self, model_weights, lm_bigrams, lm_unigrams, beam_width=5, top_k_tm=5):
        """
        model_weights: dict: {english_word: {thai_word: prob}}
        lm_bigrams: dict of counts {(w1,w2): count}
        lm_unigrams: dict of counts {w: count}
        beam_width: beam size
        top_k_tm: for each Thai token, keep top_k English candidates (speeds decoding)
        """
        self.lm_bigrams = lm_bigrams
        self.lm_unigrams = lm_unigrams
        self.beam_width = beam_width
        self.top_k_tm = top_k_tm

        # vocabulary size for Laplace smoothing in LM
        self.vocab_size = max(1, len(self.lm_unigrams))

        # Build inverted translation map: thai -> [(eng, prob), ...]
        self.inverted_tm = collections.defaultdict(list)
        for eng_word, tmap in model_weights.items():
            # CRITICAL FIX: Skip 'NULL' so it doesn't appear in the final English output
            if eng_word == "NULL":
                continue

            for thai_word, prob in tmap.items():
                # Only keep non-zero probabilities
                if prob and prob > 0.0:
                    self.inverted_tm[thai_word].append((eng_word, prob))

        # Sort candidate lists by prob descending and optionally truncate
        for t_word, cand in list(self.inverted_tm.items()):
            cand.sort(key=lambda x: x[1], reverse=True)
            self.inverted_tm[t_word] = cand[:self.top_k_tm]

        # tiny epsilon to avoid log(0)
        self._eps = 1e-12

    def _clean_thai_tokens(self, thai_text_or_list):
        if isinstance(thai_text_or_list, list):
            toks = thai_text_or_list
        else:
            # Use 'newmm' to match EMTrainer's tokenization
            toks = word_tokenize(thai_text_or_list, engine="newmm")
        
        out = []
        for t in toks:
            if t is None: continue
            t = t.replace("\u200b", "").replace("\ufeff", "").strip()
            if t != "":
                out.append(t)
        return out

    def _lm_log_prob(self, w1, w2):
        # Laplace (add-1) smoothing on counts -> P(w2|w1)
        bigram = self.lm_bigrams.get((w1, w2), 0)
        unigram = self.lm_unigrams.get(w1, 0)
        denom = unigram + self.vocab_size + 1
        prob = (bigram + 1) / denom
        return math.log(prob)

    def decode(self, thai_text):
        """
        Returns (tokens_list, score)
        tokens_list is the sequence of predicted English tokens (no <START>/<END>)
        """
        thai_tokens = self._clean_thai_tokens(thai_text)
        n = len(thai_tokens)

        # Precompute for this sentence: eng -> list of (thai_index, P(thai|eng))
        eng_to_thai = collections.defaultdict(list)
        candidate_english = set()
        
        for idx, t in enumerate(thai_tokens):
            for eng, prob in self.inverted_tm.get(t, []):
                eng_to_thai[eng].append((idx, prob))
                candidate_english.add(eng)

        # Always allow an <END> token and optionally a small set of fillers
        candidate_english.add("<END>")
        
        # Initialize beam with start token
        start = Hypothesis(["<START>"], 0.0, set())
        beam = [start]
        completed = []

        # Allow sentence to grow a bit longer than input
        max_steps = max(5, n * 3) 

        for step in range(max_steps):
            new_candidates = []

            for hyp in beam:
                last_tok = hyp.tokens[-1]

                # If hypothesis finished, keep it
                if last_tok == "<END>":
                    completed.append(hyp)
                    continue

                # If we already covered all Thai tokens, force end
                if len(hyp.covered_indices) == n:
                    next_words = ["<END>"]
                else:
                    next_words = candidate_english

                prev_word = last_tok

                for next_word in next_words:
                    # 1. LM log-prob
                    try:
                        lm_score = self._lm_log_prob(prev_word, next_word)
                    except Exception:
                        lm_score = math.log(self._eps)

                    # 2. TM log-prob: look whether this eng word translates any uncovered thai token
                    best_tm_log = float("-inf")
                    best_index = None

                    # If we have precomputed mapping for this next_word:
                    for (thai_idx, prob) in eng_to_thai.get(next_word, []):
                        if thai_idx in hyp.covered_indices:
                            continue
                        if prob <= 0:
                            continue
                        
                        logp = math.log(prob)
                        if logp > best_tm_log:
                            best_tm_log = logp
                            best_index = thai_idx

                    if best_tm_log == float("-inf"):
                        # next_word didn't explain any uncovered thai word.
                        # Give it a small TM penalty to allow function words (like 'the', 'is')
                        if next_word == "<END>":
                            tm_score = 0.0
                        else:
                            # Penalty for inserting a word that explains nothing (spontaneous generation)
                            tm_score = math.log(1e-6) 
                    else:
                        tm_score = best_tm_log

                    total_score = hyp.score + lm_score + tm_score

                    new_covered = set(hyp.covered_indices)
                    if best_index is not None:
                        new_covered.add(best_index)

                    new_hyp = Hypothesis(hyp.tokens + [next_word], total_score, new_covered)
                    new_candidates.append(new_hyp)

            if not new_candidates:
                break

            # Prune duplicates: keep top scoring hypothesis for each (last_token, covered_signature)
            bucket = {}
            for h in new_candidates:
                # Key state is: (last_word, set_of_covered_indices)
                key = (h.tokens[-1], tuple(sorted(h.covered_indices)))
                existing = bucket.get(key)
                if existing is None or h.score > existing.score:
                    bucket[key] = h

            unique_candidates = list(bucket.values())
            unique_candidates.sort(key=lambda x: x.score, reverse=True)
            beam = unique_candidates[: self.beam_width]

            # early stop if all beam entries finished
            if all(h.tokens[-1] == "<END>" for h in beam):
                completed.extend([h for h in beam if h.tokens[-1] == "<END>"])
                break

        # Choose best completed hypothesis
        if completed:
            completed.sort(key=lambda x: x.score, reverse=True)
            best = completed[0]
            # strip <START> and <END>
            tokens = [t for t in best.tokens if t not in ("<START>", "<END>")]
            return tokens, best.score

        # else return best partial
        if beam:
            best = max(beam, key=lambda x: x.score)
            tokens = [t for t in best.tokens if t != "<START>"]
            return tokens, best.score

        return [], 0.0