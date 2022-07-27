import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def openDocument(path):
    return open(path, 'r', errors = 'ignore', encoding = '#utf-8')

f = openDocument('Chatbot//DataText//PierreInfoText.txt')

def tokenization(doc):
    raw_doc = doc.read()
    raw_doc = raw_doc.lower()
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('stopwords')
    return nltk.sent_tokenize(raw_doc), nltk.word_tokenize(raw_doc)

sent_tokens, word_tokens = tokenization(f)

sent_tokens[:5]

word_tokens[:2]

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
    tokenizer=LemNormalize,
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
        robo1_response = robo1_response + sent_tokens[idx] + "\nBOT: Si je ne réponds pas correctement à votre question, essayez de la reformuler. Si vous avez une autre question n'hésitez pas à me la poser!"
        return robo1_response

def protocol(wordTokens):
    flag = True
    intro = input('BOT: Bonjour, je suis votre assistant aujourd\'hui. Comment puis-je vous aider ?\nSi vous avez un message d\'erreur, tapez "erreur", si vous voulez en savoir plus sur le monde de la pierre, tapez "info".\nTapez "sortie" à tout moment quand vouz souhaitez quitter la conversation\n')

    while intro.lower() not in ['info', 'erreur', 'sortie']:
        intro = str(input('BOT: Désolé, il semble y avoir un problème. Veuillez taper soit le mot \"info\" soit le mot \"erreur\"\n'))

    if intro.lower() == 'info':
        print('BOT: Je vous écoute! Veuillez me poser une question sur le monde du taille de la pierre.\nLe moins de mots vous inserez, le plus facile sera pour moi de vous donner une réponse pertinente :)')
        while(flag == True):
            user_response = input()
            user_response = user_response.lower()
            if(user_response != 'sortie'):
                if(user_response == 'merci' or user_response == 'merci beaucoup'):
                    flag = False
                    print('BOT: Aucun problème!')
                else:
                    sent_tokens.append(user_response)
                    wordTokens = wordTokens + nltk.word_tokenize(user_response)
                    final_words = list(set(wordTokens))
                    print("BOT: ", end = "")
                    print(response(user_response))
                    sent_tokens.remove(user_response)
            else:
                flag = False
                print("BOT: Au revoir! A bientôt")
                
    elif intro.lower() == 'erreur':
        print ('BOT: partie erreur')

    else:
        flag = False
        print("BOT: Au revoir! A bientôt")
        
protocol(word_tokens)