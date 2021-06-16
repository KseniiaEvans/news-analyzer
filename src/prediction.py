from matplotlib.pyplot import sca
from preprocessing import df
from nltk.corpus import stopwords
import pandas as pd
import re
import string
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

stop = set(stopwords.words('english'))

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return ' '.join(text.split())

def remove_stopwords(data):
    words = [word for word in data.split() if word not in stop]
    words= " ".join(words)
    return words     

def create_manual_df():
    df_manual_fake = df[df['target'] == 0].tail(10)
    df_manual_true = df[df['target'] == 1].tail(10)
    index_fake_start = df_manual_fake.index[0] -1 
    index_fake_end = df_manual_fake.index[-1]
    index_real_start = df_manual_true.index[0] -1
    index_real_end = df_manual_true.index[-1]

    for i in range(index_fake_end,index_fake_start,-1):
        df.drop([i], axis = 0, inplace = True)
        
    for i in range(index_real_end,index_real_start,-1):
        df.drop([i], axis = 0, inplace = True)

    df_manual_testing = pd.concat([df_manual_fake,df_manual_true], axis = 0, ignore_index=True)
    df_manual_testing.to_csv("../data/manual_testing.csv")
    return df_manual_testing

df_manual_testing = create_manual_df()
print('Created manual testing data')


# data preparation
df = df.drop(["title", "subject","date"], axis = 1)
df = df.sample(frac = 1)
df.reset_index(inplace = True)
df.drop(['index'], axis = 1, inplace = True)
df['text'] = df['text'].apply(wordopt)
print('Cleaned data from bad symbols')
df['text'] = df['text'].apply(remove_stopwords)
print('Removed stop words')


# Defining dependent and independent variables
x = df["text"]
y = df["target"]


# Splitting Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# Convert text to vectors
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print('Converted data to vectors')


# Don't expect good results from these classifiers, it was my first attempt ^_^
LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)
LR_score = LR.score(xv_test, y_test)
print('Trained LogisticRegression: score is {}'.format(LR_score))

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT_score = DT.score(xv_test, y_test)
print('Trained DecisionTreeClassifier: score is {}'.format(DT_score))

print('Ready to work!')

def output_lable(n):
    return 'Real' if n else 'Fake' 
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    
    return print("\nDT Prediction: {}\nLR Prediction: {}".format(
        output_lable(pred_DT[0]),
        output_lable(pred_LR[0])
        ))

def generate_manual_testing():
    for index, row in df_manual_testing.iterrows():
        print("\n\nExpected: {}".format(output_lable(row.target)))
        manual_testing(row.text)
        print('-----------------------------------------')

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def plot_top_words():
    common_words = get_top_n_bigram(df["text"], 20)
    df3 = pd.DataFrame(common_words, columns = ['bigram' , 'count'])

    fig = go.Figure([go.Bar(x=df3['bigram'], y=df3['count'])])
    fig.update_layout(title=go.layout.Title(text="Top 20 bigrams in the news text after removing stop words"))
    fig.show()