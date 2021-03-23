#Packages
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import string
import re
import json
from nltk.corpus import wordnet

#All stopwords
STOPWORDS = set(stopwords.words('english'))
engtl_stop_words = []
for words in open(r'./resources/English-Tagalog Stopwords.txt', 'r'):
    engtl_stop_words.append(words.strip().lower())

with open(r'./resources/shortcuts_dict.txt', 'r') as s:
    data = s.read()
shortcut = json.loads(data)

#clean text
def main(data, content_col):
    data["CLEANED"] = data[content_col].apply(clean_text)

    #Initiate classes
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    df = load_topwords(data, ngram=(1,3))
    return df

def substitutions(text):
    #remove all symbols except those in retain list
    retain_symbols_list = ['%', "#", "@", "_","/"]
    omit_symbols_list = [punc for punc in string.punctuation if punc not in retain_symbols_list]
    text = "".join(u for u in text if u not in omit_symbols_list)

    text = text.replace("'","")
    #replace numbered percentages with X% to represent rates etc.
    text = re.sub(r"[.a-zA-Z0-9]+%", "x%", text)
    
    # Replace all # with 'number'
    text = re.sub(r"#", " number ", text)
    
    #replace those with dates
    text = re.sub(r"(^\d{1,2}\/\d{1,2}\/\d{4}$)|(((\d{2})|(\d))\/((\d{2})|(\d))\/((\d{4})|(\d{2})))|(^[0,1]?\d{1}\/(([0-2]?\d{1})|([3][0,1]{1}))\/(([1]{1}[9]{1}[9]{1}\d{1})|([2-9]{1}\d{3}))$)", "[date]", text)

    # Replace all / with spaced character to seperate words
    text = re.sub(r"/", " ", text)   
    
    # Replace 89100 with hotline
    text = re.sub(r"(89100)|(8900)", " hotline ", text)   

    #replace those with php,p,k to [amount]
    text = re.sub(r"php([.0-9]+)|php ([.0-9]+)|p([.0-9]+)|p ([.0-9]+)|([.0-9]+)k|([.0-9]+) k|p([.0-9]+)k|php([.0-9]+)k", "[amount]", text)  
    return text

def remove_stopwords(text):
    """Remove stopwords based on nltk english word list and own list"""
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and word not in engtl_stop_words)
    return text

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# @st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def clean_text(text):
    try:          
        text = substitutions(text.strip().lower())
        text = ' '.join(shortcut[word] if word in shortcut else word for word in text.split()) 
        text = remove_stopwords(text)
        text = ' '.join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text))
        return text
    except:
        return text

#If word counts:
# def vectorize(data, ngram=(1,3)):
#         count_vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram, binary=True, min_df=2, strip_accents='unicode')
#         # count_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram,min_df=2, strip_accents='unicode')
#         count_data = count_vectorizer.fit_transform(data["CLEANED"].dropna().astype(str))
#         return count_vectorizer, count_data
 
# def get_topwords(data, count_vectorizer, count_data):
#     #Extract and pad features
#     vector = pd.DataFrame(zip(count_vectorizer.get_feature_names(), count_data.sum(0).getA1()), count_vectorizer.inverse_transform(count_data))
#     vector.columns = ['word','importance','count']
#     vector = vector.sort_values(by='importance', ascending=False)
#     return vector

#If tf-idf


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['words', 'importance']
    df = df.sort_values(by='importance', ascending=False)
    return df

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def vectorize(data, ngram=(1,3)):
        count_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram,min_df=2, strip_accents='unicode')
        count_data = count_vectorizer.fit_transform(data["CLEANED"].dropna().astype(str))
        features = count_vectorizer.get_feature_names()
        return count_data, features
 

def load_topwords(data, ngram=(1,3)):
    count_data, features = vectorize(data, ngram)
    plotdf = top_mean_feats(count_data, features)
    return plotdf