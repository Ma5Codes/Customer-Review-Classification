import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Function to detect urgency
def detect_urgency(text):
    urgency_keywords = ["urgent", "asap", "immediately", "right now", "help", "emergency"]
    return any(keyword in text.lower() for keyword in urgency_keywords)











# import re
# import string

# # Text Cleaning and Preprocessing
# def clean_text(text):
#     text = text.lower()  # Lowercase
#     text = re.sub(r"\d+", "", text)  # Remove digits
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
#     return text
