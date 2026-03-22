import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Placement Intelligence Dashboard",
    page_icon="graduation_cap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Midnight Purple + Gold
# ─────────────────────────────────────────
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0f0c29;
        color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1540;
    }

    /* Title */
    h1, h2, h3 {
        color: #f9ca24 !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1a1540;
        border: 1px solid #302b63;
        border-radius: 10px;
        padding: 15px;
    }

    div[data-testid="metric-container"] label {
        color: #ffffff !important;
    }

    div[data-testid="metric-container"] div {
        color: #f9ca24 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #f9ca24;
        color: #0f0c29;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 30px;
        font-size: 16px;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #f0932b;
        color: #0f0c29;
    }

    /* Input fields */
    .stSlider label {
        color: #f9ca24 !important;
        font-weight: bold;
    }

    .stSelectbox label {
        color: #f9ca24 !important;
        font-weight: bold;
    }

    /* Success/Error boxes */
    .success-box {
        background-color: #1a1540;
        border: 2px solid #f9ca24;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .error-box {
        background-color: #1a1540;
        border: 2px solid #f0932b;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    /* Divider */
    hr {
        border-color: #302b63;
    }

    /* DataFrame */
    .dataframe {
        background-color: #1a1540 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD & TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv('data/cleaned_data.csv')

    df['Internship_Flag'] = df['Internship_Experience'].apply(
        lambda x: 1 if str(x).strip().title() == 'Yes' else 0
    )

    feature_cols = [
        'IQ', 'Prev_Sem_Result', 'CGPA',
        'Academic_Performance', 'Extra_Curricular_Score',
        'Communication_Skills', 'Projects_Completed',
        'Internship_Flag'
    ]

    X = df[feature_cols]
    y = df['Placement_Flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=10
    )
    model.fit(X_train, y_train)

    return model, df, feature_cols


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #f9ca24; font-size: 2.5rem;'>
        Placement Intelligence Dashboard
    </h1>
    <p style='text-align: center; color: #ffffff; font-size: 1rem;'>
        BCA Final Year Project — Analyzing 10,000 Student Placement Records
    </p>
    <hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
with st.spinner('Loading model...'):
    model, df, feature_cols = load_and_train()

# ─────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────
st.sidebar.markdown("""
    <h2 style='color: #f9ca24;'>Navigation</h2>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Explorer", "Placement Predictor", "Key Insights"]
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("""
    <p style='color: #ffffff; font-size: 12px;'>
    Placement Intelligence Dashboard<br>
    BCA Final Year Project 2026<br>
    Built with Python & Streamlit
    </p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────
if page == "Overview":
    st.markdown("## Dashboard Overview")

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Students", f"{len(df):,}")
    with col2:
        placed = df[df['Placement'] == 'Placed']
        st.metric("Students Placed", f"{len(placed):,}")
    with col3:
        placement_rate = round(len(placed) / len(df) * 100, 2)
        st.metric("Placement Rate", f"{placement_rate}%")
    with col4:
        st.metric("Average CGPA", f"{df['CGPA'].mean():.2f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Two columns layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Placement Distribution")
        placement_counts = df['Placement'].value_counts()
        st.bar_chart(placement_counts)

    with col2:
        st.markdown("### CGPA Band Distribution")
        cgpa_counts = df['CGPA_Band'].value_counts()
        st.bar_chart(cgpa_counts)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Internship Experience")
        intern_counts = df['Internship_Experience'].value_counts()
        st.bar_chart(intern_counts)

    with col2:
        st.markdown("### IQ Band Distribution")
        iq_counts = df['IQ_Band'].value_counts()
        st.bar_chart(iq_counts)

# ─────────────────────────────────────────
# PAGE 2 — DATA EXPLORER
# ─────────────────────────────────────────
elif page == "Data Explorer":
    st.markdown("## Data Explorer")
    st.markdown("Filter and explore the student dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        placement_filter = st.selectbox(
            "Filter by Placement",
            ["All", "Placed", "Not Placed"]
        )
    with col2:
        cgpa_filter = st.selectbox(
            "Filter by CGPA Band",
            ["All"] + list(df['CGPA_Band'].unique())
        )
    with col3:
        intern_filter = st.selectbox(
            "Filter by Internship",
            ["All", "Yes", "No"]
        )

    filtered_df = df.copy()

    if placement_filter != "All":
        filtered_df = filtered_df[
            filtered_df['Placement'] == placement_filter
        ]
    if cgpa_filter != "All":
        filtered_df = filtered_df[
            filtered_df['CGPA_Band'] == cgpa_filter
        ]
    if intern_filter != "All":
        filtered_df = filtered_df[
            filtered_df['Internship_Experience'] == intern_filter
        ]

    st.markdown(f"Showing **{len(filtered_df):,}** students")

    display_cols = [
        'CGPA', 'IQ', 'Communication_Skills',
        'Projects_Completed', 'Internship_Experience',
        'Overall_Score', 'Placement'
    ]

    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Average CGPA by Placement")
        cgpa_avg = filtered_df.groupby('Placement')['CGPA'].mean()
        st.bar_chart(cgpa_avg)

    with col2:
        st.markdown("### Average IQ by Placement")
        iq_avg = filtered_df.groupby('Placement')['IQ'].mean()
        st.bar_chart(iq_avg)

# ─────────────────────────────────────────
# PAGE 3 — PLACEMENT PREDICTOR
# ─────────────────────────────────────────
elif page == "Placement Predictor":
    st.markdown("## Placement Predictor")
    st.markdown("Enter student details to predict placement probability")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Academic Details")
        cgpa = st.slider(
            "CGPA", min_value=0.0, max_value=10.0,
            value=7.5, step=0.1
        )
        prev_sem = st.slider(
            "Previous Semester Result",
            min_value=0, max_value=100, value=75
        )
        academic = st.slider(
            "Academic Performance",
            min_value=0.0, max_value=10.0,
            value=7.0, step=0.1
        )
        iq = st.slider(
            "IQ Score",
            min_value=50, max_value=150, value=100
        )

    with col2:
        st.markdown("### Skills & Activities")
        communication = st.slider(
            "Communication Skills",
            min_value=0.0, max_value=10.0,
            value=7.0, step=0.1
        )
        projects = st.slider(
            "Projects Completed",
            min_value=0, max_value=10, value=3
        )
        extra = st.slider(
            "Extra Curricular Score",
            min_value=0.0, max_value=10.0,
            value=6.0, step=0.1
        )
        internship = st.selectbox(
            "Internship Experience",
            ["Yes", "No"]
        )

    internship_flag = 1 if internship == "Yes" else 0

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Predict Placement"):
        student_data = pd.DataFrame([[
            iq, prev_sem, cgpa, academic,
            extra, communication, projects,
            internship_flag
        ]], columns=feature_cols)

        prediction  = model.predict(student_data)[0]
        probability = model.predict_proba(student_data)[0]
        confidence  = max(probability) * 100

        if prediction == 1:
            st.markdown(f"""
                <div class='success-box'>
                    <h2 style='color: #f9ca24;'>PLACED</h2>
                    <h3 style='color: #ffffff;'>
                        Confidence: {confidence:.1f}%
                    </h3>
                    <p style='color: #ffffff;'>
                        This student is likely to get placed
                        based on their profile!
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='error-box'>
                    <h2 style='color: #f0932b;'>NOT PLACED</h2>
                    <h3 style='color: #ffffff;'>
                        Confidence: {confidence:.1f}%
                    </h3>
                    <p style='color: #ffffff;'>
                        This student may need to improve
                        their skills to get placed.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Student Profile Summary")

        profile_data = {
            'Metric': [
                'CGPA', 'IQ', 'Communication Skills',
                'Projects', 'Prev Sem Result',
                'Extra Curricular', 'Internship'
            ],
            'Value': [
                cgpa, iq, communication,
                projects, prev_sem,
                extra, internship
            ]
        }
        st.dataframe(
            pd.DataFrame(profile_data),
            use_container_width=True
        )

# ─────────────────────────────────────────
# PAGE 4 — KEY INSIGHTS
# ─────────────────────────────────────────
elif page == "Key Insights":
    st.markdown("## Key Insights")
    st.markdown("Important findings from the analysis of 10,000 students")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Performance")
        st.markdown("""
        | Model | Accuracy |
        |---|---|
        | Logistic Regression | 90.35% |
        | Random Forest | 99.85% |
        """)

        st.markdown("### Feature Importance")
        importance_data = {
            'Feature': [
                'Communication Skills', 'IQ', 'CGPA',
                'Projects Completed', 'Prev Sem Result',
                'Extra Curricular', 'Academic Performance',
                'Internship'
            ],
            'Importance': [
                0.277, 0.264, 0.169,
                0.156, 0.114, 0.009,
                0.008, 0.002
            ]
        }
        st.bar_chart(
            pd.DataFrame(importance_data).set_index('Feature')
        )

    with col2:
        st.markdown("### Key Findings")
        st.markdown("""
        **Finding 1 — Placement is Competitive**
        Only 16.59% of students got placed out of 10,000.
        Placement requires strong overall profile.

        **Finding 2 — Communication is King**
        Communication Skills is the #1 predictor of placement
        at 27.7% importance — more than CGPA!

        **Finding 3 — CGPA Matters**
        Students with Excellent CGPA (8.5+) have 33.3%
        placement rate vs 4.4% for Below Average students.

        **Finding 4 — IQ is Critical**
        High IQ students (120+) are 5x more likely to get
        placed than Below Average IQ students.

        **Finding 5 — Internship Not Enough**
        Even students with internship experience mostly
        did not get placed — overall profile matters more.

        **Finding 6 — Projects Help**
        Projects Completed ranks 4th in importance at 15.6%
        — building projects significantly helps placement.
        """)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### Placement Rate by CGPA Band")
    cgpa_rate = df.groupby('CGPA_Band')['Placement_Flag'].mean() * 100
    st.bar_chart(cgpa_rate)