#Importing Libraries
import re
from re import sub
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import json
import string

#Functions
lemmatizer = WordNetLemmatizer()

#All stopwords
# STOPWORDS = set(stopwords.words('english'))
engtl_stop_words = []
# for words in open(r'English-Tagalog Stopwords.txt', 'r'):
for words in open(r'./resources/English-Tagalog Stopwords.txt', 'r'):
    engtl_stop_words.append(words.strip().lower())

# with open(r'shortcuts_dict.txt', 'r') as s:
with open(r'./resources/shortcuts_dict.txt', 'r') as s:
    data = s.read()
shortcut = json.loads(data)


def substitutions(text):
    #remove all symbols except those in retain list
    retain_symbols_list = ['%', "#", "@", "_","/", "!", "?"]
    omit_symbols_list = [punc for punc in string.punctuation if punc not in retain_symbols_list]
    text = "".join(u for u in text if u not in omit_symbols_list)
    
    text = text.replace("'","")
    # replace numbered percentages with X% to represent rates etc.
    text = re.sub(r"[.a-zA-Z0-9]+%", "x%", text)
    
    # Replace all # with 'number'
    text = re.sub(r"#", " number ", text)
    
    #replace those with dates
    text = re.sub(r"(^\d{1,2}\/\d{1,2}\/\d{4}$)|(((\d{2})|(\d))\/((\d{2})|(\d))\/((\d{4})|(\d{2})))|(^[0,1]?\d{1}\/(([0-2]?\d{1})|([3][0,1]{1}))\/(([1]{1}[9]{1}[9]{1}\d{1})|([2-9]{1}\d{3}))$)", "[date]", text)

    # Replace all / with spaced character to seperate words
    text = re.sub(r"/", " ", text)  
    
    # Replace all ! with spaced character to seperate words
    text = re.sub(r"!", " ! ", text)  
    
    # Replace all ! with spaced character to seperate words
    text = sub(r"\?", " ? ", text)
    
    # Replace 89100 with hotline
    text = re.sub(r"(89100)|(8900)|(1800)|(88910000)", " hotline ", text)  

    #replace those with php,p,k to [amount]
    text = re.sub(r"php([.0-9]+)|php ([.0-9]+)|p([.0-9]+)|p ([.0-9]+)|([.0-9]+)k|([.0-9]+) k|p([.0-9]+)k|php([.0-9]+)k", "[amount]", text)  

    #Detect email address
    generic_email = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    custom_email = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$' 
    text = re.sub(rf"({generic_email})|({custom_email})", " email ", text)
    return text

def remove_stopwords(text):
    """Remove stopwords based on nltk english word list and own list"""
    text = ' '.join(word for word in text.split() if word not in engtl_stop_words)
    return text

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    try:      
        text = str(text).strip().lower()
        # text = " ".join(translator.translate(x, dest='en').text if translator.detect(x).lang=='tl' else x for x in [text])
        text = substitutions(text)
        text = ' '.join(shortcut[word] if word in shortcut else word for word in text.split()) 
        text = remove_stopwords(text)
        text = ' '.join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text))
        return text.split()
    except:
        return text.split()

def clean_text1(text):
    try:      
        text = str(text).strip().lower()
        text = substitutions(text)
        text = ' '.join(shortcut[word] if word in shortcut else word for word in text.split()) 
        return text.split()
    except:
        return text.split()  

def init_clean(df, col):
    df['init_cleaned'] = df[col].map(lambda text: clean_text1(text)).str.join(" ")
    return df


def clean_text2(text):
    try:      
        text = remove_stopwords(text)
        text = ' '.join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text))
        return text.split()
    except:
        return text.split()

def final_clean(df, col):
    df['cleaned'] = df[col].map(lambda text: clean_text2(text)).str.join(" ")
    return df

def clean(df, col):
    df['cleaned'] = df[col].map(lambda text: clean_text(text)).str.join(" ")
    return df
