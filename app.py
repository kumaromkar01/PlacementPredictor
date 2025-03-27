import numpy as np
import streamlit as st
import pickle

# Load model
lg = pickle.load(open('placement.pkl', 'rb'))

# Configure page
st.set_page_config(page_title="Job Placement Predictor", page_icon="📊", layout="wide")

# Header
st.title("🎯 Job Placement Prediction Model")
st.markdown("---")

# Sidebar with instructions
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. Fill in all the fields below
    2. Click the Predict button
    """)
    st.markdown("### Feature Information:")
    st.markdown("""
    - **SSC Percentage**: Secondary School marks (%)
    - **HSC Percentage**: Higher Secondary marks (%)  
    - **Degree Percentage**: Undergraduate marks (%)
    - **Gender**: Male or Female
    - **SSC Board**: Central or Other
    - **HSC Board**: Central or Other
    - **HSC Subject**: Commerce, Science, or Arts
    - **Work Experience**: Yes or No
    - **Department**: ECE or CSE
    """)

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Candidate Details")
    
    with st.form("prediction_form"):
        # Academic Performance
        st.markdown("### Academic Performance")
        ssc_percent = st.slider("SSC Percentage (%)", 0, 100, 65)
        hsc_percent = st.slider("HSC Percentage (%)", 0, 100, 65)
        degree_percent = st.slider("Degree Percentage (%)", 0, 100, 65)
        
        # Personal Details
        st.markdown("### Personal Details")
        gender = st.radio("Gender", ["Male", "Female"])
        
        # Educational Boards
        st.markdown("### Educational Boards")
        ssc_board = st.radio("SSC Board", ["Central", "Others"])
        hsc_board = st.radio("HSC Board", ["Central", "Others"])
        
        # Subject and Experience
        st.markdown("### Subject & Experience")
        hsc_subject = st.radio("HSC Subject", ["Commerce", "Science", "Arts"])
        work_exp = st.radio("Work Experience", ["Yes", "No"])
        department = st.radio("Department", ["ECE", "CSE"])
        
        submitted = st.form_submit_button("Predict Placement")
        
        if submitted:
            # Convert all inputs to numerical values in EXACT order model expects
            input_list = [
                ssc_percent,  # ssc_percentage
                hsc_percent,   # hsc_percentage
                degree_percent,  # percentage
                1 if gender == "Male" else 0,  # gender_M
                1 if ssc_board == "Others" else 0,  # ssc_board_Others
                1 if hsc_board == "Others" else 0,  # hsc_board_Others
                1 if hsc_subject == "Commerce" else 0,  # hsc_subject_Commerce
                1 if hsc_subject == "Science" else 0,  # hsc_subject_Science
                1 if work_exp == "Yes" else 0,  # work_experience_Yes
                1 if department == "ECE" else 0   # department_ECE
            ]
            
            np_df = np.asarray(input_list, dtype=float).reshape(1, -1)
            
            try:
                prediction = lg.predict(np_df)
                proba = lg.predict_proba(np_df)[0]
                
                st.markdown("---")
                if prediction[0] == 1:
                    st.success("✅ This candidate is likely to be placed!")
                else:
                    st.warning("❌ This candidate is not likely to be placed")
                
                st.write(f"Confidence: {max(proba)*100:.1f}%")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

with col2:
    st.subheader("About the Model")
    st.markdown("""
    This model predicts job placement likelihood based on:
    - Academic performance (SSC, HSC, Degree)
    - Gender and work experience
    - Educational boards and subjects
    - Department specialization
    """)
    
    st.markdown("### Model Metrics")
    st.metric("Accuracy", "87%")
    st.metric("Precision", "85%")
    
    st.markdown("---")
    st.caption("Note: Predictions are based on historical placement data")