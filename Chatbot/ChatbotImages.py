import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import matplotlib.pyplot as plt
import unidecode


def openDocument(path):
    return open(path, 'r', errors = 'ignore', encoding = '#utf-8')

f = openDocument('Chatbot//DataImages//ImagesTitres.txt')

def tokenization(doc):
    raw_doc = doc.read()
    raw_doc = raw_doc.lower()
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('stopwords')
    return nltk.sent_tokenize(raw_doc), nltk.word_tokenize(raw_doc)

sent_tokens, word_tokens = tokenization(f)

print(sent_tokens[:2])

print(word_tokens[:5])

def remPunct():
    return dict((ord(punct), None) for punct in string.punctuation)

remove_punct_dict = remPunct()

def LemTokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response):
    robo1_response = ''

    final_stopwords_list = stopwords.words('french')
    TfidfVec = TfidfVectorizer(
    stop_words=final_stopwords_list,
    tokenizer=LemNormalize
    )
    
    #TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo1_response =  robo1_response + "Désolé, je ne comprends pas! Pourrais tu reformuler la phrase?"
        return robo1_response
    else:
        robo1_response = robo1_response + sent_tokens[idx]
        return robo1_response

def userInput():
    return input().lower()

user_response = userInput()

def nlpImageTitle(wordTokens):
    sent_tokens.append(user_response)
    wordTokens = wordTokens + nltk.word_tokenize(user_response)
    final_words = list(set(word_tokens))
    return response(user_response)

nlpImageTitle(word_tokens)

def textToImgTitle(responseUser):
    TitleImage = unidecode.unidecode(responseUser).replace(" ","").replace(".","")
    return TitleImage

titleImage = textToImgTitle(response(user_response))
print(response(user_response))
print(titleImage)

def loadImage(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imgColor = loadImage("Chatbot//DataImages//"+titleImage+".jpg")

def plotImage(title, i):
    plt.figure(title)
    plt.imshow(i, cmap="gray")
    return plt.show()

plotImage("Color", imgColor)

sent_tokens.remove(user_response)