# Fairness_Analysis_In_Recruitment_Systems
A Streamlit-based web application designed to predict hiring probabilities for job candidates while promoting fairness across gender. The system processes CVs, evaluates candidate qualifications, and provides data-driven insights through interactive dashboards.

Features
CV Upload & Processing: Upload CVs in DOCX or PDF formats to extract key details (name, skills, experience, etc.).
Role-Based Evaluation: Predict hiring probabilities for roles like Software Engineer, Game Developer, Web Developer, and Data Scientist.
Fairness Adjustment: Adjusts predictions to mitigate gender bias, ensuring equitable evaluations.
Interactive Dashboard: Visualizes average hiring probabilities by role and gender distribution.
Data Viewer & EDA: Displays candidate data in a table and provides exploratory data analysis with downloadable Excel output.
Candidate Shortlisting: Automatically shortlists the top 5 candidates per role based on hiring probability.
Tech Stack
Python: Core programming language.
Streamlit: For the web interface.
Pandas & NumPy: Data manipulation and analysis.
Scikit-learn: Random Forest Classifier for predictions.
Matplotlib: Data visualization.
python-docx & pdfplumber: CV parsing.
Pickle: Data caching.
