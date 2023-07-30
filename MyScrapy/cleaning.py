import re
import nltk
lemmatize = nltk.WordNetLemmatizer()
def clean_text_parse(i):
    text = i[:-8]
    text = re.sub("[^a-zA-Z]", " ", text)
    text = "".join([word.lower() for word in text])
    text = nltk.word_tokenize(text, language="english")
    text = [lemmatize.lemmatize(word) for word in text]
    text = " ".join(text)
    return text

def clean_text_request(i):
    text = re.sub("[^a-zA-Z]", " ", i)
    text = "".join([word.lower() for word in text])
    text = nltk.word_tokenize(text, language = "english")
    text = [lemmatize.lemmatize(word) for word in text]
    text = " ".join(text)
    return text

