# Resume Screening App 

This is a simple web application that screens and categorizes resumes using Natural Language Processing (NLP) techniques. It allows users to upload resume files in either .txt or .pdf format, and it predicts the category or job role associated with the uploaded resume.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)

## Overview

1. **Training Notebook**: A Jupyter Notebook (`resume_screening.ipynb`) for preprocessing and training a machine learning model using scikit-learn and NLTK. It includes the following steps:
   - Data preprocessing (cleaning, encoding labels, removing stopwords, and lemmatization).
   - Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency).
   - Training a multi-class classifier (OneVsRestClassifier) based on K-Nearest Neighbors (KNN).
   - Saving the trained model and TF-IDF vectorizer for future use.


2. **Streamlit Web App**: A Streamlit web application (`app.py`) for user interaction and predictions. Users can upload their resumes, and the app will predict the category or job role associated with the uploaded resume.

## Getting Started

### Prerequisites

Before running the code, ensure that you have the following prerequisites installed:


- Required Python packages (install them using `pip install -r requirements.txt`)



## Usage

### Training Notebook

*(if custom training required then go with the training step else run the app.py)*

1. Open the `resume_screening.ipynb` notebook using Jupyter Notebook.

2. Follow the code in the notebook to preprocess the resume dataset, train the machine learning model, and save the trained model and TF-IDF vectorizer.

### Streamlit Web App

1. Run the Streamlit web app using the following command:

   ```bash
   streamlit run app.py
   ```

2. Access the app in your web browser by following the link provided in the terminal.

3. Upload a resume (in .txt or .pdf format) to the app.

4. The app will process the resume and predict the associated category or job role.
