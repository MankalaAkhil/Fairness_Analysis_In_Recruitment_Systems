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
