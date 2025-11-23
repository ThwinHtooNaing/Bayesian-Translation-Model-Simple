import streamlit as st
import os
import time
import math
from pythainlp.tokenize import word_tokenize

# --- IMPORT TRANSLATOR BACKEND ---
# We use the filename 'bayesian_translator' (lowercase) containing the class 'BayesianTranslator'
try:
    from BayesianTranslator import BayesianTranslator
except ImportError:
    st.error("Could not import 'BayesianTranslator'. Make sure 'bayesian_translator.py' is in the same folder.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Bayesian Thai-En Translator",
    page_icon="🇹🇭",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 20px !important;
    }
    .success-box {
        padding: 20px;
        background-color: #f0fdf4;
        border-radius: 10px;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🇹🇭 ➡️ 🇬🇧 Bayesian Translator")
st.markdown("""
This system uses a **Statistical Machine Translation (SMT)** pipeline:

- **Translation Model:** IBM Model-1 / Model-2  
- **Language Model:** Bigram LM  
- **Decoder:** Stack-based Beam Search

It finds the most probable English sentence given Thai input using the formula:
$$ \hat{e} = \\text{argmax}_e \ P(e) \cdot P(f|e) $$
""")
st.divider()

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    model = BayesianTranslator()
    pkl_file = "translator_model.pkl"
    csv_file = "nus_sms.csv"

    # 1. Try loading pre-trained pickle
    if os.path.exists(pkl_file):
        success = model.load_model(pkl_file)
        if success:
            return model, None
    
    # 2. Fallback: Train from CSV if pickle missing
    if os.path.exists(csv_file):
        # We need to import trainer here only if we need to train
        try:
            from trainer import EMTrainer
            
            status_text = "Model file not found. Training from scratch (this may take a moment)..."
            # Return a special status to show spinner in main thread
            return None, "TRAIN_NEEDED"
            
        except ImportError:
            return None, "Missing 'trainer.py'. Cannot train model."
            
    return None, "No model file ('translator_model.pkl') and no training data ('nus_sms.csv') found."

# Handle the loading/training logic with UI feedback
model = None
temp_model, status = load_model()

if status == "TRAIN_NEEDED":
    with st.spinner("Training Translation Model & Language Model..."):
        # Re-instantiate and train
        from trainer import EMTrainer
        model = BayesianTranslator()
        csv_file = "nus_sms.csv"
        
        # Train EM
        em_trainer = EMTrainer()
        em_trainer.load_data(csv_file)
        em_trainer.initialize_uniform()
        em_trainer.train_model1(iterations=10)
        em_trainer.train_model2(iterations=5)
        model.translation_table = em_trainer.t
        
        # Train LM
        model.train_lm_from_csv(csv_file)
        
        # Save for next time
        model.save_model("translator_model.pkl")
        st.success("Training Complete! Model saved.")
elif status:
    st.error(f"🚨 Error: {status}")
    st.stop()
else:
    model = temp_model

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Thai Input")
    thai_text = st.text_area(
        "Enter Thai text:",
        value="ฉันรักคุณ",
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Model Info")
    st.write(f"**Vocab Size:** {len(model.lm_unigrams):,}")
    st.write(f"**Bigrams:** {len(model.lm_bigrams):,}")
    st.write(f"**Total Words:** {model.total_words:,}")
    st.write("**Beam Width:** 10") # Matches backend default

# --- TRANSLATE BUTTON ---
if st.button("Translate", type="primary", use_container_width=True):

    if not thai_text.strip():
        st.warning("⚠️ Please enter some Thai text.")
        st.stop()

    # STEP 1 — SHOW TOKENIZATION
    # Using the helper from the class ensures consistency with backend
    tokens = model.tokenize_thai(thai_text)
    
    with st.expander("🔍 Internal Thai Tokenization", expanded=False):
        st.write("The sentence is split into these units for translation:")
        st.code(tokens, language="python")

    # STEP 2 — PERFORM TRANSLATION
    start_time = time.time()
    best_trans, log_score = model.translate(thai_text)
    end_time = time.time()

    st.divider()

    # STEP 3 — RESULTS
    st.subheader("English Output")
    if not best_trans:
        st.warning("No translation found. The words might be outside the vocabulary.")
    else:
        st.markdown(
            f'<div class="success-box">{best_trans}</div>',
            unsafe_allow_html=True
        )
        # st.caption(f"Raw: {repr(best_trans)}")

    # STEP 4 — METRICS
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Log Probability", f"{log_score:.2f}")
    with m2:
        # Protect against overflow if score is very close to 0
        try:
            linear_prob = math.exp(log_score)
            st.metric("Linear Probability", f"{linear_prob:.2e}")
        except OverflowError:
            st.metric("Linear Probability", "0.00")
    with m3:
        st.metric("Time", f"{end_time - start_time:.3f}s")

    # STEP 5 — EXPLANATION
    st.info(f"""
    ### Understanding the Score
    The decoder maximizes **P(English | Thai)**. 
    - The score is the **Log Probability** of the sentence.
    - **{log_score:.2f}** indicates the cumulative likelihood of this path in the beam search.
    - A value closer to **0** (less negative) is mathematically "better".
    """)