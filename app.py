import streamlit as st
import pickle
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional for WordNet Lemmatizer

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Mapping of encoded labels to categories
label_to_category = {
    0: 'Advocate', 1: 'Arts', 2: 'Automation Testing',
    3: 'Blockchain', 4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science',
    7: 'Database', 8: 'DevOps Engineer', 9: 'DotNet Developer', 10: 'ETL Developer',
    11: 'Electrical Engineering', 12: 'HR', 13: 'Hadoop', 14: 'Health and fitness',
    15: 'Java Developer', 16: 'Mechanical Engineer', 17: 'Network Security Engineer',
    18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer',
    21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'
}

# Load the trained classifier and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean the resume text
def clean_resume(text):
    pattern = r'[^a-zA-Z0-9\s]'
    cleantext = re.sub(pattern, '', text)
    cleantext = re.sub(r'\n|\r', '', cleantext)
    return cleantext

# Function to remove stopwords from the text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to lemmatize the text
def lemmatize(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Streamlit app main function
def main():
    st.title('Resume Screening App')
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if upload_file is not None:
        # Show a warning note about potential inaccuracies with PDF files
        st.warning("Note: The PDF format resume predictions might be inaccurate due to encoding and decoding issues.")

        # Read and decode the uploaded file
        resume_bytes = upload_file.read()
        try:
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        # Clean, preprocess, and lemmatize the resume text
        cleaned_resume_re = clean_resume(resume_text)
        cleaned_resume_sw = remove_stopwords(cleaned_resume_re)
        lemmatized_resume = lemmatize(cleaned_resume_sw)

        # Transform the lemmatized resume text using the TF-IDF vectorizer
        final_resume = tfidf.transform([lemmatized_resume])

        # Make predictions using the loaded classifier
        prediction = clf.predict(final_resume)[0]

        # Get the predicted category and display it
        final_prediction = label_to_category.get(prediction, 'unknown')
        st.write(final_prediction)

if __name__ == '__main__':
    main()
