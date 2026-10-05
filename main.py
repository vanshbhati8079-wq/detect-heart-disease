import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# ====================== Page Configuration ======================
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"   # We are using centered layout
)

# Custom CSS for Beautiful UI
st.markdown("""
<style>
    .main {padding: 2rem;}
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 25px 0;
    }
    .high-risk { 
        background: linear-gradient(135deg, #ffe6e6, #ffcccc); 
        border: 4px solid #ff4b4b;
    }
    .low-risk { 
        background: linear-gradient(135deg, #e6f7e6, #d4edda); 
        border: 4px solid #28a745;
    }
    .stButton>button {
        width: 100%;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ====================== Load Model & Scaler ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.keras", compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ====================== Title ======================
st.title("❤️ HeartGuard AI")
st.markdown('<p class="subtitle">Advanced Heart Disease Risk Prediction System</p>', unsafe_allow_html=True)

# ====================== Sample Buttons ======================
st.subheader("Quick Test Samples")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔴 Load High Risk Sample", use_container_width=True):
        st.session_state.update({
            "age": 63, "sex": "Male", "cp": "Asymptomatic", "trestbps": 145,
            "chol": 233, "fbs": "Yes", "restecg": "Normal", "thalach": 150,
            "exang": "Yes", "oldpeak": 2.3, "slope": "Flat", "ca": 2, "thal": "Reversible Defect"
        })

with col2:
    if st.button("🟢 Load Low Risk Sample", use_container_width=True):
        st.session_state.update({
            "age": 40, "sex": "Female", "cp": "Non-anginal Pain", "trestbps": 120,
            "chol": 180, "fbs": "No", "restecg": "Normal", "thalach": 170,
            "exang": "No", "oldpeak": 0.0, "slope": "Upsloping", "ca": 0, "thal": "Normal"
        })

st.divider()

# ====================== Input Section ======================
st.subheader("Patient Information")

col_a, col_b = st.columns(2)

with col_a:
    age = st.slider("Age", 20, 80, value=st.session_state.get("age", 55), key="age")
    
    sex = st.radio("Sex", ["Female", "Male"], 
                   index=0 if st.session_state.get("sex", "Male") == "Female" else 1,
                   key="sex", horizontal=True)
    sex_num = 0 if sex == "Female" else 1

    cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp_label = st.selectbox("Chest Pain Type", list(cp_dict.keys()), 
                            index=list(cp_dict.keys()).index(st.session_state.get("cp", "Asymptomatic")), 
                            key="cp")
    cp = cp_dict[cp_label]

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 
                         value=st.session_state.get("trestbps", 130), key="trestbps")
    
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 
                     value=st.session_state.get("chol", 220), key="chol")

with col_b:
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], 
                   index=0 if st.session_state.get("fbs", "No") == "No" else 1, 
                   key="fbs", horizontal=True)
    fbs_num = 1 if fbs == "Yes" else 0

    restecg_dict = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
    restecg_label = st.selectbox("Resting Electrocardiographic Results", 
                                 list(restecg_dict.keys()), 
                                 index=list(restecg_dict.keys()).index(st.session_state.get("restecg", "Normal")), 
                                 key="restecg")
    restecg = restecg_dict[restecg_label]

    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 
                        value=st.session_state.get("thalach", 160), key="thalach")

    exang = st.radio("Exercise Induced Angina?", ["No", "Yes"], 
                     index=0 if st.session_state.get("exang", "No") == "No" else 1, 
                     key="exang", horizontal=True)
    exang_num = 1 if exang == "Yes" else 0

# Second Row
col_c, col_d = st.columns(2)

with col_c:
    oldpeak = st.slider("Oldpeak (ST Depression induced by exercise)", 0.0, 6.0, 
                        value=st.session_state.get("oldpeak", 1.0), step=0.1, key="oldpeak")

    slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_label = st.selectbox("Slope of the Peak Exercise ST Segment", 
                               list(slope_dict.keys()), 
                               index=list(slope_dict.keys()).index(st.session_state.get("slope", "Flat")), 
                               key="slope")
    slope = slope_dict[slope_label]

with col_d:
    ca = st.slider("Number of Major Vessels colored by Fluoroscopy (0-3)", 0, 3, 
                   value=st.session_state.get("ca", 0), key="ca")

    thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal_label = st.selectbox("Thalassemia", list(thal_dict.keys()), 
                              index=list(thal_dict.keys()).index(st.session_state.get("thal", "Reversible Defect")), 
                              key="thal")
    thal = thal_dict[thal_label]

# ====================== Predict Button ======================
if st.button("🚀 Predict Heart Disease Risk", type="primary", use_container_width=True):
    
    # Prepare data
    input_data = np.array([[age, sex_num, cp, trestbps, chol, fbs_num,
                            restecg, thalach, exang_num, oldpeak, slope, ca, thal]])
    
    scaled_input = scaler.transform(input_data)
    
    # Make Prediction
    prediction = model.predict(scaled_input, verbose=0)
    prob = float(prediction[0][0])
    risk_percent = prob * 100

    # ====================== Show Result ======================
    st.subheader("🔮 Prediction Result")

    if prob > 0.5:
        st.markdown(f"""
        <div class="result-card high-risk">
            <h2>⚠️ High Risk of Heart Disease</h2>
            <h1 style="color: #ff4b4b; font-size: 4.8rem;">{risk_percent:.1f}%</h1>
            <p><strong>Immediate medical consultation is strongly recommended.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.error("🔴 High Risk Detected")
    else:
        st.markdown(f"""
        <div class="result-card low-risk">
            <h2>✅ Low Risk of Heart Disease</h2>
            <h1 style="color: #28a745; font-size: 4.8rem;">{risk_percent:.1f}%</h1>
            <p><strong>Good news! Risk level appears to be low.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.success("🟢 Low Risk")

    st.progress(risk_percent / 100)
    st.metric(label="Heart Disease Probability", value=f"{risk_percent:.1f}%")

# ====================== About Section ======================
st.divider()
st.subheader("About HeartGuard AI")
st.write("""
This application uses a deep learning neural network (trained on the Cleveland Heart Disease Dataset) 
to predict the likelihood of heart disease based on 13 important clinical parameters.
""")

st.caption("Note: This is for educational and demonstration purposes only.")
