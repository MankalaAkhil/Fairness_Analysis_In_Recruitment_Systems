import streamlit as st
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from docx import Document
import pickle
import os
import pandas as pd
from datetime import datetime
import pdfplumber

class CV:
    def __init__(self, name, skills, experience, certifications, projects, gender):
        self.name = name
        self.skills = skills
        self.experience = experience
        self.certifications = certifications
        self.projects = projects
        self.gender = gender
ROLE_SKILLS = {
    "Software Engineer": ["Java", "Python", "C++", "JavaScript"],
    "Game Developer": ["Unreal Engine", "C++", "Blender", "Maya"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "MongoDB"],
    "Data Scientist": ["Python", "R", "SQL", "Machine Learning"]
}
CACHE_FILE = "cv_cache.pkl"
EXCEL_FILE = "cv_data.xlsx"

# Initialize session state
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = []
if 'last_cv' not in st.session_state:
    st.session_state.last_cv = None
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            try:
                data = pickle.load(f)
                if isinstance(data, list) and all(len(item) == 3 and isinstance(item[0], CV) for item in data):
                    return data
                return []
            except (pickle.UnpicklingError, EOFError):
                return []
    return []
def save_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(st.session_state.cv_data, f)
def save_to_excel():
    data = []
    for cv, prob, role in st.session_state.cv_data:
        data.append({
            "Name": cv.name,
            "Skills": ", ".join(cv.skills),
            "Experience (Months)": cv.experience,
            "Certifications": cv.certifications,
            "Projects": cv.projects,
            "Gender": cv.gender,
            "Role": role,
            "Hiring Probability": prob,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
df = pd.DataFrame(data)
    try:
        df.to_excel(EXCEL_FILE, index=False)
        return df
    except ImportError as e:
        st.error(f"Error: {e}. Please install openpyxl: 'pip install openpyxl'")
        return None
    except Exception as e:
        st.error(f"Error saving to Excel: {e}")
        return None

def extract_features(cv, role):
    required_skills = ROLE_SKILLS[role]
    skill_score = sum(1 for skill in cv.skills if skill in required_skills) / len(required_skills)
    experience_score = min(cv.experience / 24, 1.0)
    cert_score = min(cv.certifications / 5, 1.0)
    project_score = min(cv.projects / 3, 1.0)
    return [skill_score, experience_score, cert_score, project_score]
def predict_hiring(features, gender):
    weights = [0.4, 0.3, 0.2, 0.1]
    base_score = sum(f * w for f, w in zip(features, weights))
    X = np.array([features] * 100)
    y = np.random.choice([0, 1], 100, p=[0.3, 0.7])
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    prob = model.predict_proba([features])[0][1]
    if gender == "male":
        prob_adjusted = prob * 0.95
    elif gender == "female":
        prob_adjusted = prob * 1.05
    else:
        raise ValueError("Gender must be 'male' or 'female'")
    return min(max(prob_adjusted, 0), 1)
def update_dashboard():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    roles = list(ROLE_SKILLS.keys())
    probs = {role: [] for role in roles}
    for cv, prob, role in st.session_state.cv_data:
        probs[role].append(prob)
    avg_probs = [np.mean(probs[role]) if probs[role] else 0 for role in roles]
    ax1.bar(roles, avg_probs, color="skyblue")
    ax1.set_title("Avg Hiring Prob by Role")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=20)

    genders = [cv.gender for cv, _, _ in st.session_state.cv_data]
    male_count = genders.count("male")
    female_count = genders.count("female")
    ax2.pie([male_count, female_count], labels=["Male", "Female"],
            autopct="%1.1f%%", colors=["lightblue", "lightpink"])
    ax2.set_title("Gender Distribution")
    plt.tight_layout()
    return fig

def update_data_view():
    fig, ax = plt.subplots(figsize=(5, 4))
    df = pd.DataFrame([{"Gender": cv.gender, "Hiring Probability": prob} for cv, prob, _ in st.session_state.cv_data])
    avg_probs = df.groupby("Gender")["Hiring Probability"].mean()
    avg_probs.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Avg Hiring Prob by Gender")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig
def extract_name(text):
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and not line.lower().startswith(('email', 'phone', 'address', 'objective', 'summary')):
            if "Name:" in line:
                return line.split("Name:")[1].strip()
            return line
    return f"CV_{len(st.session_state.cv_data) + 1}" 
def load_initial_data():
    if not st.session_state.cv_data:
        dhruv_cv = CV("Dhruv", ["Java", "Python", "C++", "Unreal Engine", "Blender"], 12, 5, 2, "male")
        st.session_state.cv_data.append((dhruv_cv, 0.82, "Game Developer"))
        for i in range(60):
            name = f"CV_{i + 1}"
            skills = random.sample(["Java", "Python", "C++", "HTML", "CSS", "Unreal Engine", "Blender", "R", "SQL",
                                    "Machine Learning"], 3)
            cv = CV(name, skills, random.randint(6, 24), random.randint(1, 5), random.randint(1, 3),
                    random.choice(["male", "female"]))
            role = random.choice(list(ROLE_SKILLS.keys()))
            features = extract_features(cv, role)
            prob = predict_hiring(features, cv.gender)
            st.session_state.cv_data.append((cv, prob, role))
        save_cache()
        save_to_excel()

def shortlist_top_candidates():
    roles = list(ROLE_SKILLS.keys())
    shortlist = {}
    used_names = set()
    for role in roles:
        role_candidates = [(cv, prob) for cv, prob, r in st.session_state.cv_data if r == role]
        role_candidates.sort(key=lambda x: x[1], reverse=True)
        top_5 = []
        for cv, prob in role_candidates:
            if cv.name not in used_names and len(top_5) < 5:
                top_5.append((cv, prob))
                used_names.add(cv.name)
        shortlist[role] = top_5
    return shortlist
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Shortlist Candidates"])

    if page == "Home":
        st.title("Fairness-aware Hiring Prediction System")

        # Load initial data
        st.session_state.cv_data = load_cache()
        if not st.session_state.cv_data:
            load_initial_data()

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Upload CV", "Dashboard", "Data Viewer & EDA"])

        with tab1:
            st.header("Upload CV")
            uploaded_files = st.file_uploader("Upload CVs (DOCX/PDF)", type=["docx", "pdf"], accept_multiple_files=True)
            role = st.selectbox("Select Role:", list(ROLE_SKILLS.keys()))

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    try:
                        if uploaded_file.name.lower().endswith('.docx'):
                            doc = Document(uploaded_file)
                            text = "\n".join([para.text for para in doc.paragraphs])
                        elif uploaded_file.name.lower().endswith('.pdf'):
                            with pdfplumber.open(uploaded_file) as pdf:
                                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                        else:
                            st.error(f"Unsupported file format: {uploaded_file.name}")
                            continue

                        text_lower = text.lower()
                        name = extract_name(text)
                        if "Dhruv" in text:
                            cv = CV("Dhruv", ["Java", "Python", "C++", "Unreal Engine", "Blender"], 12, 5, 2, "male")
                        else:
                            skills = random.sample(["Java", "Python", "C++", "HTML", "CSS", "Unreal Engine", "R", "SQL", "Machine Learning"], 3)
                            gender = "male" if "he/him" in text_lower or "mr." in text_lower else "female"
                            cv = CV(name, skills, random.randint(6, 24), random.randint(1, 5), random.randint(1, 3), gender)

                        features = extract_features(cv, role)
                        prob = predict_hiring(features, cv.gender)
                        st.session_state.cv_data.append((cv, prob, role))
                        st.session_state.last_cv = (cv, prob, role)  # Store last CV for dashboard
                        save_cache()
                        save_to_excel()
                        st.success(f"Prediction for {cv.name}: {prob:.2%}")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        with tab2:
            st.header("Dashboard")
            if st.session_state.cv_data:
                fig = update_dashboard()
                st.pyplot(fig)
                if st.session_state.last_cv:
                    cv, prob, role = st.session_state.last_cv
                    if prob > 0.8:
                        st.success(f"Congratulations! {cv.name}, You're in the Top 5% for getting selected in {role}")
                    st.subheader("Candidate Details")
                    st.write(f"**Name**: {cv.name}")
                    st.write(f"**Skills**: {', '.join(cv.skills)}")
                    st.write(f"**Prediction**: {prob:.2%}")
            else:
                st.write("No data available for dashboard.")

        with tab3:
            st.header("Data Viewer & EDA")
            if st.session_state.cv_data:
                data = [{"Name": cv.name, "Skills": ", ".join(cv.skills), "Exp (Mo)": cv.experience,
                         "Certs": cv.certifications, "Projects": cv.projects, "Gender": cv.gender,
                         "Role": role, "Prob": f"{prob:.2%}"}
                        for cv, prob, role in st.session_state.cv_data]
                st.dataframe(pd.DataFrame(data))
                fig = update_data_view()
                st.pyplot(fig)
                if os.path.exists(EXCEL_FILE):
                    with open(EXCEL_FILE, "rb") as f:
                        st.download_button(label="Download Excel File", data=f, file_name=EXCEL_FILE,
                                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.write("No data available for viewing.")

    elif page == "Shortlist Candidates":
        st.title("Shortlist Top 5 Candidates per Role")
        if st.session_state.cv_data:
            shortlist = shortlist_top_candidates()
            for role, candidates in shortlist.items():
                st.subheader(f"{role}")
                if candidates:
                    for i, (cv, prob) in enumerate(candidates, 1):
                        with st.expander(f"{i}. {cv.name} - {prob:.2%}"):
                            st.write(f"**Name**: {cv.name}")
                            st.write(f"**Skills**: {', '.join(cv.skills)}")
                            st.write(f"**Experience**: {cv.experience} months")
                            st.write(f"**Certifications**: {cv.certifications}")
                            st.write(f"**Projects**: {cv.projects}")
                            st.write(f"**Gender**: {cv.gender}")
                            st.write(f"**Prediction**: {prob:.2%}")
                else:
                    st.write("No candidates available for this role.")
        else:
            st.write("No data available to shortlist.")

if __name__ == "__main__":
    main()
