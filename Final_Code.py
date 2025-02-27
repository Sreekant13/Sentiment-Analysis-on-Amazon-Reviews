import pandas as pd 
import numpy as np 
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup 
import gzip
import requests
from io import BytesIO, TextIOWrapper
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Storing URL of the dataset in url
url = "https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz"

# Here we will fetch the dataset
response = requests.get(url)
response.raise_for_status()  #  Had to use this to ensure that the request was successful

# Now we need to decompress the content to use it for further tasks
with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
    amazon_review_data = pd.read_csv(
        TextIOWrapper(gz, encoding='utf-8'),  # we set this explicitly to UTF-8 encoding because it was giving error for some values which was not in the proper format
        sep='\t',
        on_bad_lines='skip',  # Here we will skip problematic lines
        low_memory=False      # Lastly we do this to improve performance for large datasets
    )


# A small function to assign star rating to binary labels
def assign_label(star_rating):
    if star_rating > 3:
        return 1
    elif star_rating <= 2:
        return 0
    else:
        return None

# we will filter relevant columns and will keep only Reviews and Ratings columns and drop the rest of the columns
amazon_review_data = amazon_review_data[['review_body', 'star_rating']].dropna()

# We need to make sure that star_rating is numeric for further parts in code
amazon_review_data['star_rating'] = pd.to_numeric(amazon_review_data['star_rating'], errors='coerce')

# Print statistics for the three classes: positive, negative, and neutral reviews
positive_count = (amazon_review_data['star_rating'] > 3).sum()
negative_count = (amazon_review_data['star_rating'] <= 2).sum()
neutral_count = (amazon_review_data['star_rating'] == 3).sum()

# Print the counts for each class in the requested format
print("-" * 50)
print(f"Positive reviews: {positive_count}, Negative reviews: {negative_count}, Neutral reviews: {neutral_count}")
print("-" * 50)

# Now we map ratings to binary labels
amazon_review_data['label'] = amazon_review_data['star_rating'].apply(assign_label)

# As sid in the assignment, we need to drop neutral reviews
amazon_review_data = amazon_review_data.dropna(subset=['label'])

# We will downsample the dataset to 100,000 positive and negative reviews according to the assignment
# Also we will be using random state = 42 such that we can get the consistent results
positive_reviews = amazon_review_data[amazon_review_data['label'] == 1].sample(100000, random_state=42) 
negative_reviews = amazon_review_data[amazon_review_data['label'] == 0].sample(100000, random_state=42)

# We will now combine the downsampled amazon_review_data
balanced_data = pd.concat([positive_reviews, negative_reviews])

# Finally the main task where we will split into training and testing datasets
train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=42)

# Lastly, we print dataset statistics as asked
print("Training size:", len(train_data))
print("Testing size:", len(test_data))
print("Class distribution in training:", train_data['label'].value_counts())
print("-" * 50)


#####DATA CLEANING#####

#Manually created contraction dictionary
contractions_dict = {
    "can't": "cannot", "won't": "will not", "i'm": "i am",
    "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "mightn't": "might not", "mustn't": "must not",
    "let's": "let us", "that's": "that is", "what's": "what is",
    "who's": "who is", "there's": "there is", "here's": "here is",
    "how's": "how is", "where's": "where is", "why's": "why is",
    "when's": "when is", "weren't": "were not", "could've": "could have",
    "should've": "should have", "would've": "would have", "might've": "might have",
    "must've": "must have", "we've": "we have", "you've": "you have",
    "they've": "they have", "who've": "who have", "i've": "i have",
    "hasn't": "has not", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "it'll": "it will", "we'll": "we will",
    "they'll": "they will", "i'll": "i will", "that'll": "that will",
    "there'll": "there will", "who'll": "who will", "what'll": "what will",
    "won't": "will not", "shan't": "shall not", "who'd": "who would",
    "it'd": "it would", "we'd": "we would", "they'd": "they would",
    "you'd": "you would", "she'd": "she would", "he'd": "he would",
    "i'd": "i would", "they're": "they are", "we're": "we are",
    "you're": "you are", "i'm": "i am", "he's": "he is",
    "she's": "she is", "it's": "it is", "ain't": "is not",
    "y'all": "you all", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "lemme": "let me", "gimme": "give me",
    "dunno": "do not know", "outta": "out of", "sorta": "sort of",
    "kinda": "kind of", "oughta": "ought to", "coulda": "could have",
    "woulda": "would have", "shoulda": "should have", "how'd": "how did",
    "why'd": "why did", "where'd": "where did", "when'd": "when did",
    "y'know": "you know", "c'mon": "come on", "how're": "how are",
    "what're": "what are", "who're": "who are", "where're": "where are",
    "when're": "when are", "why're": "why are", "there're": "there are",
    "that'd": "that would", "this'll": "this will", "it'll've": "it will have",
    "we'll've": "we will have", "who'll've": "who will have", 
    "it'd've": "it would have", "nothin'": "nothing", "somethin'": "something",
    "everythin'": "everything", "givin'": "giving", "movin'": "moving",
    "y'all've": "you all have", "y'all'd": "you all would", 
    "ain'tcha": "are not you", "didn'tcha": "did not you",
    "ya'll": "you all", "ain'tcha": "are not you", "mightn't've": "might not have",
    "mustn't've": "must not have", "shouldn't've": "should not have",
    "you'd've": "you would have", "there'd've": "there would have",
    "who'd've": "who would have", "what'd've": "what would have"
}

# Custom Function to expand contractions based on whatever is needed
def expand_contractions(text):
    words = text.split()
    expanded_words = []
    for word in words:
        # We will check if the word is in the contractions dictionary
        if word in contractions_dict:
            # Here we replace the word with its expanded form
            expanded_words.append(contractions_dict[word])
        else:
            # Keep the word as is
            expanded_words.append(word)
    return " ".join(expanded_words)

# Cleaning function for the dataset cleaning
def clean_text(text):
    text = text.lower()  # We will convert text to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # We will Remove URLs
    text = re.sub(r'<.*?>', '', text)  # We will remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # We will remove non-alphabetical characters
    text = re.sub(r'\s+', ' ', text).strip()  # We will remove extra spaces
    text = expand_contractions(text)  # We will expand contractions manually
    return text

# Apply cleaning to training and testing datasets
train_data['cleaned_review'] = train_data['review_body'].apply(clean_text)
test_data['cleaned_review'] = test_data['review_body'].apply(clean_text)

# Print average length before and after cleaning
print("Average length before cleaning:", train_data['review_body'].str.len().mean())
print("Average length after cleaning:", train_data['cleaned_review'].str.len().mean())
print("-" * 50)

#####Pre-Processing#####

###Remove Stop Words:###
from nltk.corpus import stopwords # type: ignore

# We will set and define stop words
stop_words = set(stopwords.words('english'))

# This function is to remove stop words
def remove_stop_words(text):
    tokens = word_tokenize(text)  # Here we will tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Here we will filter out stop words
    return tokens

###Perform Lemmanization:###
from nltk.stem import WordNetLemmatizer # type: ignore
# We will initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to perform lemmatization
def lemmatize_tokens(tokens):
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]  # Here we will lemmatize each token
    return ' '.join(lemmatized)  # Here we will join tokens back into a string

# We need to ensure that all values in 'cleaned_review' are strings and replace NaN values with an empty string
train_data['cleaned_review'] = train_data['cleaned_review'].fillna("").astype(str)
test_data['cleaned_review'] = test_data['cleaned_review'].fillna("").astype(str)

# This is the preprocessing function where we will call the other functions(like the remove stop word function and lemmatize token function) for preprocessing our data
def preprocess_text(text):
    tokens = remove_stop_words(text)  # Remove stop words
    processed_text = lemmatize_tokens(tokens)  # Perform lemmatization
    return processed_text

# Apply preprocessing to the dataset
train_data['preprocessed_review'] = train_data['cleaned_review'].apply(preprocess_text)
test_data['preprocessed_review'] = test_data['cleaned_review'].apply(preprocess_text)

# Print three random samples before and after preprocessing
print("Three random samples before and after preprocessing:")
sample_indices = random.sample(range(len(train_data)), 3)  # Select 3 random indices
for i in sample_indices:
    print(f"Sample {i + 1}:")
    print("Before preprocessing:", train_data['cleaned_review'].iloc[i])
    print("After preprocessing:", train_data['preprocessed_review'].iloc[i])
    print("-" * 50)
    
# Handle NaN values in 'preprocessed_review'
train_data['preprocessed_review'] = train_data['preprocessed_review'].fillna("")
test_data['preprocessed_review'] = test_data['preprocessed_review'].fillna("")

print("Average length after data cleaning and before preprocessing:", train_data['cleaned_review'].str.len().mean())
print("Average length after data cleaning and after preprocessing:", train_data['preprocessed_review'].str.len().mean())
print("-" * 50)

###### TF-IDF Feature Extraction #####

# We will do feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_data['preprocessed_review']).toarray()
X_test = tfidf.transform(test_data['preprocessed_review']).toarray()

y_train = train_data['label']
y_test = test_data['label']

# This function is to train and evaluate a model(Made it as a function for simlpicity purpose)
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)  # We train the model here
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # After trainging we will calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Finally we will print metrics in required format
    print(f"Accuracy_train: {train_accuracy:.4f}, Precision_train: {train_precision:.4f}, Recall_train: {train_recall:.4f}, F1_train: {train_f1:.4f}, Accuracy_test: {test_accuracy:.4f}, Precision_test: {test_precision:.4f}, Recall_test: {test_recall:.4f}, F1_test: {test_f1:.4f}")
    print("-" * 20)

# Evaluate each model separately
print("Perceptron:")
print("-" * 20)
evaluate_model(Perceptron(), "Perceptron")

print("LinearSVC:")
print("-" * 20)
evaluate_model(LinearSVC(), "LinearSVC")

print("Logistic Regression:")
print("-" * 20)
evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression")

print("Naive Bayes:")
print("-" * 20)
evaluate_model(MultinomialNB(), "Naive Bayes")
