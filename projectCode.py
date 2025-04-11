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
