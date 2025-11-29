import streamlit as st
import pandas as pd
import json
import os
import time

try:
    from trainer import ThaiToEngTrainer
    from stack_decoder import StackDecoder, SimpleLanguageModel
except ImportError as e:
    st.error(f"CRITICAL ERROR: Missing modules. {e}")
    st.info("Please ensure 'trainer.py' and 'stack_decoder.py' are in the same directory as this app.")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Bayesian Translator",
    page_icon="🇹🇭",
    layout="wide"
)

# --- Session State Management ---
if 'decoder' not in st.session_state:
    st.session_state['decoder'] = None
if 'lm' not in st.session_state:
    st.session_state['lm'] = None
if 'system_ready' not in st.session_state:
    st.session_state['system_ready'] = False
if 'active_data_path' not in st.session_state:
    st.session_state['active_data_path'] = "nus_sms.csv"  # Default

# --- Helper Functions ---
def save_uploaded_file(uploaded_file):
    """Saves uploaded file to disk so the Trainer class can read it path-wise"""
    try:
        file_path = "temp_custom_data.csv"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def load_system(data_path):
    """Loads the weights and initializes the decoder using specific data for LM"""
    model_path = "model_weights_th_en.json"

    if not os.path.exists(model_path):
        return False, "Model weights not found. Please train the model first."
    
    if not os.path.exists(data_path):
        return False, f"Dataset ({data_path}) not found."

    # 1. Load TM
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            tm_weights = data.get("translation", {})
    except Exception as e:
        return False, f"Error loading JSON weights: {e}"

    # 2. Train LM (Language Model needs the corpus to build n-grams)
    try:
        lm = SimpleLanguageModel()
        lm.train(data_path)
        st.session_state['lm'] = lm
    except Exception as e:
        return False, f"Error initializing Language Model with {data_path}: {e}"

    # 3. Init Decoder
    st.session_state['decoder'] = StackDecoder(
        model_weights=tm_weights,
        lm_bigrams=lm.bigrams,
        lm_unigrams=lm.unigrams,
        beam_width=20,
        top_k_tm=10
    )
    
    st.session_state['system_ready'] = True
    return True, f"System Loaded Successfully using {data_path}!"

def train_system(iterations_m1, iterations_m2, data_path):
    """Runs the EM training pipeline on the specific data path"""
    if not os.path.exists(data_path):
        return False, f"Dataset {data_path} not found."

    try:
        trainer = ThaiToEngTrainer()
        # Ensure your trainer.py accepts a path in load_data
        trainer.load_data(data_path) 
        trainer.initialize_uniform()
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Model 1 Training
        total_steps = iterations_m1 + iterations_m2
        current_step = 0

        status_text.text("Training IBM Model 1 (Lexical)...")
        trainer.train_model1(iterations=iterations_m1)
        current_step += iterations_m1
        progress_bar.progress(current_step / total_steps)

        # Model 2 Training
        status_text.text("Training IBM Model 2 (Alignment)...")
        trainer.train_model2(iterations=iterations_m2)
        progress_bar.progress(1.0)
        
        trainer.save_model("model_weights_th_en.json")
        status_text.text("Training Complete. Saved to model_weights_th_en.json")
        return True, "Training finished!"
    except Exception as e:
        return False, f"Training Error: {e}"

# --- Sidebar UI ---
with st.sidebar:
    st.title("Control Panel")

    # --- Dataset Selection Section ---
    st.subheader("📁 Dataset Configuration")
    data_source = st.radio(
        "Select Training Data:",
        ("Default (nus_sms.csv)", "Upload Custom CSV")
    )

    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV (Must have 'Thai' and 'English' cols)", type=['csv'])
        if uploaded_file is not None:
            saved_path = save_uploaded_file(uploaded_file)
            if saved_path:
                st.session_state['active_data_path'] = saved_path
                st.success(f"Using: {uploaded_file.name}")
        else:
            st.warning("Please upload a file.")
            # Fallback prevents crash if they switch mode but don't upload
            if st.session_state['active_data_path'] == "nus_sms.csv":
                st.session_state['active_data_path'] = "nus_sms.csv"
    else:
        st.session_state['active_data_path'] = "nus_sms.csv"

    st.info(f"Active Dataset: `{st.session_state['active_data_path']}`")

    st.markdown("---")
    
    # --- Status Section ---
    st.subheader("System Status")
    if st.session_state['system_ready']:
        st.success("🟢 System Ready")
    else:
        st.warning("🔴 Not Loaded")

    st.markdown("---")
    
    # --- Training Controls ---
    st.subheader("Training Controls")
    iter_m1 = st.number_input("Model 1 Iterations", min_value=1, value=10)
    iter_m2 = st.number_input("Model 2 Iterations", min_value=1, value=5)
    
    if st.button("Train New Model", type="primary"):
        with st.spinner(f"Training on {st.session_state['active_data_path']}..."):
            success, msg = train_system(iter_m1, iter_m2, st.session_state['active_data_path'])
            if success:
                st.success(msg)
                # Auto-load after training
                load_system(st.session_state['active_data_path'])
                time.sleep(1) # Give user time to read success msg
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    if st.button("Reload System"):
        success, msg = load_system(st.session_state['active_data_path'])
        if success:
            st.success(msg)
            time.sleep(1)
            st.rerun()
        else:
            st.error(msg)

# --- Main UI ---
st.title("🇹🇭 Bayesian Thai-English Translator")
st.markdown("Using **IBM Model 2** (Likelihood) + **Bigram Language Model** (Prior) + **Multi-Stack Decoding**.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Translate", "Model Weights", "Current Dataset"])

# TAB 1: TRANSLATION
with tab1:
    if not st.session_state['system_ready']:
        st.info(" Please **Train** or **Load** the system from the sidebar to start.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input (Thai)")
            thai_input = st.text_area("Enter Thai text here:", height=150, placeholder="สวัสดีครับ...")
            
            if st.button("Translate", type="primary", use_container_width=True):
                if thai_input.strip():
                    with st.spinner("Decoding..."):
                        start_time = time.time()
                        tokens, score = st.session_state['decoder'].decode(thai_input)
                        end_time = time.time()
                        
                        result_text = " ".join(tokens)
                        
                        st.session_state['last_result'] = result_text
                        st.session_state['last_score'] = score
                        st.session_state['last_time'] = end_time - start_time
                else:
                    st.warning("Please enter some text.")

        with col2:
            st.subheader("Output (English)")
            if 'last_result' in st.session_state:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                    <h3 style="margin:0; color: #333;">{st.session_state['last_result']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Metrics")
                m1, m2 = st.columns(2)
                m1.metric("Log Probability", f"{st.session_state['last_score']:.2f}")
                m2.metric("Decoding Time", f"{st.session_state['last_time']:.3f}s")
            else:
                st.markdown("*Translation will appear here.*")

# TAB 2: WEIGHTS
with tab2:
    st.header("Inspect Learned Weights")
    
    if not os.path.exists("model_weights_th_en.json"):
        st.warning("No model file found.")
    else:
        try:
            with open("model_weights_th_en.json", "r", encoding="utf-8") as f:
                model_data = json.load(f)
                tm = model_data.get("translation", {})
            
            # Convert to DataFrame for display
            flat_data = []
            for thai_word, eng_map in tm.items():
                for eng_word, prob in eng_map.items():
                    flat_data.append({"Thai": thai_word, "English": eng_word, "Probability": prob})
            
            df_weights = pd.DataFrame(flat_data)
            
            # Filter
            search_term = st.text_input("Search for a Thai word:", "")
            if search_term:
                df_weights = df_weights[df_weights['Thai'].str.contains(search_term, na=False)]
            
            st.dataframe(
                df_weights.sort_values(by="Probability", ascending=False), 
                use_container_width=True,
                height=500
            )
            
        except Exception as e:
            st.error(f"Error reading model file: {e}")

# TAB 3: DATASET
with tab3:
    current_path = st.session_state.get('active_data_path', 'nus_sms.csv')
    st.header(f"Dataset Preview: {current_path}")
    
    if os.path.exists(current_path):
        try:
            df = pd.read_csv(current_path)
            st.dataframe(df, use_container_width=True)
            st.write(f"**Total sentences:** {len(df)}")
            
            # Validation Check
            cols = [c.lower() for c in df.columns]
            # Assumes trainer expects specific columns, usually implicit or index based, 
            # but good to visually check.
            st.caption("Note: Ensure your CSV has columns that match what 'trainer.py' expects (usually 'Thai' and 'English').")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.warning(f"File not found: {current_path}")