import json
import os
import time

try:
    from trainer import ThaiToEngTrainer
    from stack_decoder import StackDecoder, SimpleLanguageModel
except ImportError as e:
    print("CRITICAL ERROR: Missing modules.")
    print(f"Details: {e}")
    print("Please ensure 'trainer.py' and 'stack_decoder.py' are in the same directory.")
    exit(1)

class BayesianTranslator:
    def __init__(self, data_file="nus_sms.csv", model_file="model_weights_th_en.json"):
        self.data_file = data_file
        self.model_file = model_file
        self.decoder = None
        self.lm = None
        
    def train_system(self, iterations_m1=10, iterations_m2=5):
        """
        Step 1: Train the Translation Model (P(Thai|Eng)) using EM Algorithm.
        Step 2: Train the Language Model (P(Eng)) using N-grams.
        """
        print("="*40)
        print("PHASE 1: Learning Translation Model (P(F|E))")
        print("="*40)
        
        # 1. Initialize and Train IBM Models
        trainer = ThaiToEngTrainer()
        
        if not os.path.exists(self.data_file):
            print(f"Error: Dataset '{self.data_file}' not found.")
            return

        trainer.load_data(self.data_file)
        trainer.initialize_uniform()
        
        # Expectation-Maximization loop
        start_time = time.time()
        trainer.train_model1(iterations=iterations_m1)
        trainer.train_model2(iterations=iterations_m2)
        print(f"Training finished in {time.time() - start_time:.2f} seconds.")

        trainer.save_model(self.model_file)
        print(f"Translation probabilities saved to {self.model_file}")

    def load_system(self):
       
        print("\n" + "="*40)
        print("PHASE 2: Initializing Bayesian Decoder")
        print("="*40)

        # 1. Load Translation Model (Likelihood)
        if not os.path.exists(self.model_file):
            print(f"Model weights {self.model_file} not found. Please train first.")
            return False

        print("Loading TM weights (Likelihood)...")
        with open(self.model_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            tm_weights = data.get("translation", {})

        # 2. Train/Load Language Model (Prior)
        print("Training LM (Prior) from corpus...")
        self.lm = SimpleLanguageModel()
        self.lm.train(self.data_file)
       
        print("Initializing Stack Decoder...")
        self.decoder = StackDecoder(
            model_weights=tm_weights,
            lm_bigrams=self.lm.bigrams,
            lm_unigrams=self.lm.unigrams,
            beam_width=20, 
            top_k_tm=10    
        )
        
        return True

    def translate(self, thai_text):
        
        if not self.decoder:
            print("Error: System not loaded.")
            return "SYSTEM_ERROR"

       
        tokens, score = self.decoder.decode(thai_text)
        
        translation = " ".join(tokens)
        
        return translation, score

# if __name__ == "__main__":
#     import sys

#     bayes_sys = BayesianTranslator(data_file="nus_sms.csv")

#     if len(sys.argv) > 1 and sys.argv[1] == "--train":
#         bayes_sys.train_system()
#     elif not os.path.exists("model_weights_th_en.json"):
#         print("Model weights not found. Starting training automatically...")
#         bayes_sys.train_system()
#     success = bayes_sys.load_system()
    
#     if success:
#         print("\n" + "="*40)
#         print("      BAYESIAN TRANSLATOR READY      ")
#         print("      (Thai -> English)              ")
#         print("="*40)
#         print("Type 'quit' to exit.\n")

#         while True:
#             user_input = input("Enter Thai sentence: ")
#             if user_input.lower() in ["quit", "exit", "q"]:
#                 break
            
#             if not user_input.strip():
#                 continue

#             trans, log_prob = bayes_sys.translate(user_input)
            
#             print(f"Translation : {trans}")
#             print(f"Log Prob    : {log_prob:.4f}")
#             print("-" * 30)