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
