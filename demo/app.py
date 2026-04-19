"""Streamlit demo application for password strength evaluation."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import EntropyBasedModel
from src.utils import (
    anonymize_output,
    calculate_entropy,
    detect_keyboard_patterns,
    detect_repeated_patterns,
    detect_sequential_patterns,
    estimate_time_to_crack,
    format_time_duration,
    hash_password,
    validate_password_input,
)


# Page configuration
st.set_page_config(
    page_title="Password Strength Evaluation",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str) -> Optional[EntropyBasedModel]:
    """Load the trained model."""
    try:
        model = EntropyBasedModel(None)
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def analyze_password(password: str) -> Dict:
    """Analyze a single password and return comprehensive results."""
    # Validate input
    is_valid, error_msg = validate_password_input(password)
    if not is_valid:
        return {"error": error_msg}
    
    # Basic analysis
    analysis = {
        "password": password,
        "length": len(password),
        "entropy": calculate_entropy(password),
        "time_to_crack": estimate_time_to_crack(password),
        "patterns": {
            "keyboard": detect_keyboard_patterns(password),
            "sequential": detect_sequential_patterns(password),
            "repeated": detect_repeated_patterns(password)
        }
    }
    
    # Character analysis
    analysis["char_analysis"] = {
        "lowercase": sum(1 for c in password if c.islower()),
        "uppercase": sum(1 for c in password if c.isupper()),
        "digits": sum(1 for c in password if c.isdigit()),
        "symbols": sum(1 for c in password if not c.isalnum()),
        "unique_chars": len(set(password))
    }
    
    # Calculate diversity
    if len(password) > 0:
        analysis["diversity"] = analysis["char_analysis"]["unique_chars"] / len(password)
    else:
        analysis["diversity"] = 0
    
    return analysis


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Password Strength Evaluation System</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research & Educational Purpose Only</h4>
    <p>This tool is designed for security research and educational purposes only. 
    It should not be used for production security operations or exploitation. 
    Results may be inaccurate and should not be the sole basis for security decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="outputs/trained_model.pkl",
        help="Path to the trained model file"
    )
    
    # Load model
    model = load_model(model_path)
    
    if model is None:
        st.error("Please train a model first using the training script.")
        st.info("Run: `python scripts/train.py`")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Single Analysis", "Batch Analysis", "Model Insights", "About"])
    
    with tab1:
        st.header("Single Password Analysis")
        
        # Password input
        password = st.text_input(
            "Enter password to analyze:",
            type="password",
            help="Enter a password to analyze its strength"
        )
        
        if password:
            # Analyze password
            analysis = analyze_password(password)
            
            if "error" in analysis:
                st.error(f"Error: {analysis['error']}")
            else:
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Length", analysis["length"])
                    st.metric("Entropy", f"{analysis['entropy']:.2f} bits")
                
                with col2:
                    st.metric("Character Diversity", f"{analysis['diversity']:.2f}")
                    st.metric("Unique Characters", analysis["char_analysis"]["unique_chars"])
                
                with col3:
                    st.metric("Time to Crack", format_time_duration(analysis["time_to_crack"]))
                    st.metric("Pattern Count", len(analysis["patterns"]["keyboard"]) + 
                             len(analysis["patterns"]["sequential"]) + 
                             len(analysis["patterns"]["repeated"]))
                
                # Character breakdown
                st.subheader("Character Analysis")
                char_data = analysis["char_analysis"]
                
                char_df = pd.DataFrame([
                    {"Type": "Lowercase", "Count": char_data["lowercase"]},
                    {"Type": "Uppercase", "Count": char_data["uppercase"]},
                    {"Type": "Digits", "Count": char_data["digits"]},
                    {"Type": "Symbols", "Count": char_data["symbols"]}
                ])
                
                fig = px.bar(char_df, x="Type", y="Count", 
                           title="Character Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Pattern analysis
                st.subheader("Pattern Analysis")
                
                patterns_found = []
                for pattern_type, patterns in analysis["patterns"].items():
                    if patterns:
                        patterns_found.extend([f"{pattern_type}: {pattern}" for pattern in patterns])
                
                if patterns_found:
                    st.warning("⚠️ Weak patterns detected:")
                    for pattern in patterns_found:
                        st.write(f"• {pattern}")
                else:
                    st.success("✅ No weak patterns detected")
                
                # Model prediction
                if model and model.is_fitted:
                    st.subheader("Model Prediction")
                    
                    try:
                        prediction = model.predict([password])[0]
                        probabilities = model.predict_proba([password])[0]
                        
                        # Display prediction
                        if prediction == "weak":
                            st.markdown('<div class="danger-box"><h4>🔴 Weak Password</h4></div>', 
                                      unsafe_allow_html=True)
                        elif prediction == "moderate":
                            st.markdown('<div class="warning-box"><h4>🟡 Moderate Password</h4></div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-box"><h4>🟢 Strong Password</h4></div>', 
                                      unsafe_allow_html=True)
                        
                        # Probability breakdown
                        prob_df = pd.DataFrame([
                            {"Strength": "Weak", "Probability": probabilities[0]},
                            {"Strength": "Moderate", "Probability": probabilities[1]},
                            {"Strength": "Strong", "Probability": probabilities[2]}
                        ])
                        
                        fig = px.bar(prob_df, x="Strength", y="Probability",
                                   title="Prediction Confidence",
                                   color="Strength",
                                   color_discrete_map={
                                       "Weak": "red",
                                       "Moderate": "orange", 
                                       "Strong": "green"
                                   })
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature explanation
                        explanation = model.explain_prediction(password, top_k=5)
                        st.subheader("Top Contributing Features")
                        
                        for feature, importance in explanation["top_features"]:
                            st.write(f"• **{feature}**: {importance:.3f}")
                    
                    except Exception as e:
                        st.error(f"Model prediction failed: {e}")
    
    with tab2:
        st.header("Batch Password Analysis")
        
        # Batch input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload CSV file", "Paste passwords"]
        )
        
        passwords_to_analyze = []
        
        if input_method == "Upload CSV file":
            uploaded_file = st.file_uploader(
                "Upload CSV file with passwords",
                type="csv",
                help="CSV file should have a 'password' column"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if "password" in df.columns:
                        passwords_to_analyze = df["password"].tolist()
                        st.success(f"Loaded {len(passwords_to_analyze)} passwords")
                    else:
                        st.error("CSV file must contain a 'password' column")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        
        else:  # Paste passwords
            password_text = st.text_area(
                "Enter passwords (one per line):",
                height=200,
                help="Enter passwords separated by newlines"
            )
            
            if password_text:
                passwords_to_analyze = [pwd.strip() for pwd in password_text.split('\n') if pwd.strip()]
                st.info(f"Found {len(passwords_to_analyze)} passwords")
        
        # Analyze batch
        if passwords_to_analyze and st.button("Analyze Passwords"):
            st.subheader("Batch Analysis Results")
            
            # Limit batch size for performance
            max_batch_size = 100
            if len(passwords_to_analyze) > max_batch_size:
                st.warning(f"Analyzing first {max_batch_size} passwords for performance")
                passwords_to_analyze = passwords_to_analyze[:max_batch_size]
            
            # Analyze passwords
            results = []
            for password in passwords_to_analyze:
                analysis = analyze_password(password)
                if "error" not in analysis:
                    results.append(analysis)
            
            if results:
                # Create results DataFrame
                results_df = pd.DataFrame([
                    {
                        "Password": anonymize_output(r["password"], 10),
                        "Length": r["length"],
                        "Entropy": r["entropy"],
                        "Diversity": r["diversity"],
                        "Time to Crack": format_time_duration(r["time_to_crack"]),
                        "Pattern Count": len(r["patterns"]["keyboard"]) + 
                                       len(r["patterns"]["sequential"]) + 
                                       len(r["patterns"]["repeated"])
                    }
                    for r in results
                ])
                
                # Display results
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Length", f"{results_df['Length'].mean():.1f}")
                    st.metric("Average Entropy", f"{results_df['Entropy'].mean():.2f}")
                
                with col2:
                    st.metric("Average Diversity", f"{results_df['Diversity'].mean():.2f}")
                    st.metric("Max Pattern Count", results_df['Pattern Count'].max())
                
                with col3:
                    st.metric("Total Passwords", len(results_df))
                    st.metric("Unique Passwords", results_df['Password'].nunique())
                
                # Distribution plots
                st.subheader("Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(results_df, x="Length", 
                                     title="Password Length Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(results_df, x="Entropy", 
                                     title="Entropy Distribution")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Insights")
        
        if model and model.is_fitted:
            # Feature importance
            st.subheader("Feature Importance")
            
            try:
                feature_importance = model.get_feature_importance()
                
                # Create feature importance DataFrame
                fi_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                
                # Display top features
                st.dataframe(fi_df.head(15), use_container_width=True)
                
                # Feature importance plot
                fig = px.bar(fi_df.head(15), x="Importance", y="Feature",
                           orientation="h", title="Top 15 Feature Importance")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to get feature importance: {e}")
            
            # Model configuration
            st.subheader("Model Configuration")
            
            if hasattr(model, 'config') and model.config:
                config_dict = OmegaConf.to_container(model.config, resolve=True)
                
                # Display configuration in expandable sections
                with st.expander("Entropy Thresholds"):
                    st.json(config_dict.get("entropy_thresholds", {}))
                
                with st.expander("Complexity Weights"):
                    st.json(config_dict.get("complexity_weights", {}))
                
                with st.expander("Classification Settings"):
                    st.json(config_dict.get("classification", {}))
        
        else:
            st.error("No trained model available")
    
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ## Password Strength Evaluation System
        
        This system provides comprehensive password strength analysis using:
        
        ### Features
        - **Entropy Analysis**: Shannon entropy calculation
        - **Pattern Detection**: Keyboard, sequential, and repeated patterns
        - **Character Analysis**: Diversity and composition analysis
        - **Time to Crack**: Brute force estimation
        - **Machine Learning**: Trained model for strength classification
        
        ### Security Metrics
        - Length and character diversity
        - Entropy-based scoring
        - Pattern recognition
        - Breach simulation
        - Time-to-crack estimation
        
        ### Model Architecture
        - Entropy-based feature extraction
        - Random Forest classifier
        - Cross-validation evaluation
        - Feature importance analysis
        
        ### Privacy & Safety
        - Password hashing for privacy
        - Input validation and sanitization
        - Anonymized output display
        - Research-only purpose
        
        ### Technical Stack
        - Python 3.10+
        - scikit-learn for ML
        - Streamlit for UI
        - Plotly for visualizations
        - Pandas for data processing
        
        ### Usage
        1. **Single Analysis**: Enter a password for detailed analysis
        2. **Batch Analysis**: Upload CSV or paste multiple passwords
        3. **Model Insights**: View feature importance and configuration
        4. **About**: Learn about the system
        
        ### Disclaimer
        This tool is for educational and research purposes only. 
        It should not be used for production security operations.
        Results may be inaccurate and should not be the sole basis 
        for security decisions.
        """)


if __name__ == "__main__":
    main()
