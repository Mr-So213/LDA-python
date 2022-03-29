# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:15:46 2022

@author: So
"""

import pandas as pd
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
import re
import os
import seaborn as sns
from gensim.models.coherencemodel import CoherenceModel
import numpy as np

#creat functions
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADV
    

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags if get_wordnet_pos(t[1]) == wordnet.NOUN]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

def extract_BM(text):
    text = text.lower()
    sentences = nltk.sent_tokenize(text)
    word = "business model"
    sentences_list = [sent for sent in sentences if word in sent]
    result = "".join(sentences_list)
    return result


#data cleaning
#clean "\n"
cleaned_before = [re.sub('\n', " ", i) for i in text_before]
cleaned_after = [re.sub('\n', " ", i) for i in text_after]

#clean "- "
cleaned_before = [re.sub('- ', "", i) for i in cleaned_before]
cleaned_after = [re.sub('- ', "", i) for i in cleaned_after]

#set exclude words
exclude_words = "|".join([ " and", " et", " al", " co", " ad", " bop", " kt", " would", " ps", " amit", " zott", " pp"])

#clean_text
cleaned_before = [clean_text(i) for i in cleaned_before]
cleaned_after = [clean_text(i) for i in cleaned_after]

#clean exclude words
cleaned_before = [re.sub(exclude_words, "", i) for i in cleaned_before]
cleaned_after = [re.sub(exclude_words, "", i) for i in cleaned_after]


#LDA
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

def make_corpus(data_words):
    corpus = []
    for text in data_words:
        id2word = corpora.Dictionary(data_words)
        new = id2word.doc2bow(text)
        corpus.append(new)
    return corpus


def LDA(lst, file_name, ntopics):
    file_name = file_name
    lemmatized_texts = lst
    data_words = gen_words(lemmatized_texts)
    corpus = make_corpus(data_words)
    id2word = corpora.Dictionary(data_words)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=ntopics,
                                               random_state=0,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha="auto")
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds")
    pyLDAvis.save_html(vis, file_name+ ".html")
    print("[0]: lda_model\n[1]: corpus\n[2]: id2word")
    return (lda_model, corpus, id2word)

def LDAnoVis(lst, ntopics):
    lemmatized_texts = lst
    data_words = gen_words(lemmatized_texts)
    corpus = make_corpus(data_words)
    id2word = corpora.Dictionary(data_words)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=ntopics,
                                               random_state=0,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha="auto")
    return (lda_model, corpus, id2word)


def CoherenceLDA(data, iteration):
    Colst = []
    times = 0
    for i in range(1, iteration + 1):
        res = LDAnoVis(data, i)
        cm = CoherenceModel(
            model=res[0],
            corpus=res[1],
            dictionary=res[2],
            coherence='u_mass', # Coherenceの算出方法を指定。 (デフォルトは'c_v')
            topn=20 # 各トピックの上位何単語から算出するか指定(デフォルト20)
            )
        Colst.append(cm.get_coherence())
        times = times + 1
        print(str(times) + "/" + str(iteration))
    
    df = pd.DataFrame()
    df["N_of_Topic"] = list(range(1, iteration + 1))
    df["Coherence"] = Colst
    return df

def PerplexityLDA(data, iteration):
    Perlst = []
    times = 0
    for i in range(1, iteration + 1):
        res = LDAnoVis(data, i)
        per = np.exp(-res[0].log_perplexity(res[1]))
        Perlst.append(per)
        times = times + 1
        print(str(times) + "/" + str(iteration))
    
    df = pd.DataFrame()
    df["N_of_Topic"] = list(range(1, iteration + 1))
    df["Perplexity"] = Perlst
    return df


#Calculate Coherence of Topics
CoDF_B = CoherenceLDA(cleaned_before, 50)
CoDF_A = CoherenceLDA(cleaned_after, 50)

#Calculate Perplexity of Topics
PerDF_B = PerplexityLDA(cleaned_before, 50)
PerDF_A = PerplexityLDA(cleaned_after, 50)

print(max(list(CoDF_B.Coherence)))
print(max(list(CoDF_A.Coherence)))

CoFig_B = sns.lineplot(data=CoDF_B, x='N_of_Topic', y='Coherence')
CoFig_A = sns.lineplot(data=CoDF_A, x='N_of_Topic', y='Coherence')

PerFig_B = sns.lineplot(data=PerDF_B, x='N_of_Topic', y='Perplexity')
PerFig_A = sns.lineplot(data=PerDF_A, x='N_of_Topic', y='Perplexity')

os.chdir(r"")

CP_A = pd.merge(CoDF_A, PerDF_A, on="N_of_Topic")

#Normalization By MAX-MIN
def min_max(l):
    l_min = min(l)
    l_max = max(l)
    return [(i - l_min) / (l_max - l_min) for i in l]

CP_A["Normal_Coh"] = min_max(CP_A["Coherence"])
CP_A["Normal_Per"] = min_max(CP_A["Perplexity"])

CP_A.to_excel("CP_A.xlsx")

#LDA comit

BM_A = LDA(cleaned_after, "BM_after2011", 6)

def gen_dfwords(lda, ntopics):
    twords = []
    rows = []
    for i in range(0,ntopics):
        rows.append("Topic" + str(i+1))
        twords.append([ word.split('"')[1] for word in lda.print_topic(i).split('+') ])
   
    dfwords = pd.DataFrame(twords, index=rows)
    return dfwords
    

def gen_dfByLDA(lda, tcorpus, ntopics):
    topics = []
    for c in tcorpus:
        topic = [0] * ntopics
        for (tpc, prob) in lda.get_document_topics(c):
            topic[tpc] = prob
        topics.append(topic)
    
    clusters = []
    for x in topics:
        clusters.append(x.index(max(x)))
    
    return [topics, clusters]
    
df_A = pd.DataFrame()

df_A["Article"] = namelst_A

TC_A = gen_dfByLDA(BM_A[0], BM_A[1], 6)
df_A["Topics"] = TC_A[1]
df_A.to_excel("Topics_A.xlsx")




    
