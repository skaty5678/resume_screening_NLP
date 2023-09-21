import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load the resume dataset
raw_df = pd.read_csv('/content/drive/MyDrive/resume_screening_NLP/UpdatedResumeDataSet.csv')

# Create a copy of the dataset
df = raw_df.copy()

# Function to clean the text in the 'Resume' column
def cleanResume(text):
    pattern = r'[^a-zA-Z0-9\s]'
    cleantext = re.sub(pattern, '', text)
    cleantext = re.sub(r'\n|\r', '', cleantext)
    return cleantext

# Apply the cleanResume function to the 'Resume' column
df['Resume'] = df['Resume'].apply(cleanResume)

# Initialize a LabelEncoder to encode 'Category'
le = LabelEncoder()

# Fit and transform 'Category' column
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

# Create a mapping of encoded labels to category names
label_to_category = dict(zip(le.transform(le.classes_), le.classes_))

# Download NLTK stopwords data
nltk.download('stopwords')

# Function to remove stopwords from text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply the remove_stopwords function to the 'Resume' column
df['Resume'] = df['Resume'].apply(remove_stopwords)

# Initialize a WordNetLemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()

# Function to lemmatize text
def lemmatize(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Apply the lemmatize function to the 'Resume' column
df['Resume'] = df['Resume'].apply(lemmatize)

# Initialize a TfidfVectorizer for text feature extraction
tfidf = TfidfVectorizer()

# Fit and transform 'Resume' column to create TF-IDF features
tfidf.fit(df['Resume'])
tfidf_matrix = tfidf.transform(df['Resume'])

# Get feature names (words) from TF-IDF vectorizer
feature_names = tfidf.get_feature_names_out()

# Create a DataFrame of TF-IDF features
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['Category'], test_size=0.2, random_state=24)

# Initialize a KNeighborsClassifier within a OneVsRestClassifier for multi-class classification
clf = OneVsRestClassifier(KNeighborsClassifier())

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Generate and print a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Save the TF-IDF vectorizer and classifier to files
tfidf_path = '/content/drive/MyDrive/resume_screening_NLP/tfidf.pkl'
pickle.dump(tfidf, open(tfidf_path, "wb"))

model_path = '/content/drive/MyDrive/resume_screening_NLP/clf.pkl'
pickle.dump(clf, open(model_path, 'wb'))

# Predict a category for a sample resume
resume_text = '''.'''
my_resume = re.sub(r'\s{2,}', ' ', resume_text)
my_resume = cleanResume(my_resume)
my_resume = remove_stopwords(my_resume)
my_resume = lemmatize(my_resume)

clf = pickle.load(open('/content/drive/MyDrive/resume_screening_NLP/clf.pkl', 'rb'))

input_features = tfidf.transform([my_resume])
prediction = clf.predict(input_features)[0]
final_prediction = label_to_category.get(prediction, 'unknown')

# Print the final predicted category
print("Predicted Category:", final_prediction)
