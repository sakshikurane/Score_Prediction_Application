import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for shadow and clean layout styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open('knn.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header Card
st.markdown("""
    <div class="card">
        <h1>🎓 Student Performance Prediction App</h1>
        <p>This application uses a K-Nearest Neighbors (KNN) regression model to predict student outcomes based on study habits, demographics, and lifestyle factors[cite: 1].</p>
    </div>
""", unsafe_allow_html=True)

# Input Form with Structured Layout
with st.form("prediction_form"):
    st.markdown("### Enter Student Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=50, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course", ["Computer Science", "Engineering", "Business", "Arts", "Science"])
        study_hours = st.number_input("Study Hours (Daily)", min_value=0.0, max_value=24.0, value=4.0)
        
    with col2:
        class_attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, value=85)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        sleep_hours = st.number_input("Sleep Hours (Daily)", min_value=0.0, max_value=24.0, value=7.0)
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"])
        
    with col3:
        study_method = st.selectbox("Study Method", ["Self-study", "Group study", "Online courses", "Hybrid"])
        facility_rating = st.slider("Facility Rating (1-10)", min_value=1, max_value=10, value=7)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Moderate", "Hard"])

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    # Note: Adjust these mappings if your training pipeline used specific LabelEncoders or OneHotEncoders.
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    internet_map = {"No": 0, "Yes": 1}
    sleep_quality_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
    exam_diff_map = {"Easy": 1, "Moderate": 2, "Hard": 3}
    
    # Basic numeric encoding placeholder for categorical variables
    input_data = np.array([[
        age,
        gender_map.get(gender, 0),
        hash(course) % 10,
        study_hours,
        class_attendance,
        internet_map.get(internet_access, 0),
        sleep_hours,
        sleep_quality_map.get(sleep_quality, 2),
        hash(study_method) % 10,
        facility_rating,
        exam_diff_map.get(exam_difficulty, 2)
    ]])

    try:
        prediction = model.predict(input_data)
        st.markdown(f"""
            <div class="card" style="text-align: center; background-color: #e8f5e9;">
                <h2 style="color: #2e7d32; margin-bottom: 0px;">Predicted Score / Result</h2>
                <p style="font-size: 2.5rem; font-weight: bold; color: #1b5e20; margin-top: 5px;">{prediction[0]:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {e}. Please check if your feature encoding matches the format expected by the model.")
