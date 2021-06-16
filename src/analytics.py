import warnings
warnings.simplefilter("ignore", UserWarning)
from preprocessing import df
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import pandas as pd
import plotly.express as px

# df['target']=0 #fake
# df['target']=1 #real

def plot_pie(savefig=True):
    plt.figure(figsize=(15,7))
    labels=['fake news','real news']
    colors = ["SkyBlue","PeachPuff"]
    plt.pie(df['target'].value_counts(),labels=labels,colors=colors,
            autopct='%1.2f%%', startangle=140) 
    if savefig: 
        plt.savefig('../pics/pie.png')
    else:
        plt.show()


def plot_subject_countplot(real=True, both=False, savefig=True):
    sub_df = df[df['target'] == int(real)] if not both else df
    title = 'Fake&Real' if both else ('Real' if real else 'Fake')
    plt.figure(figsize=(15,7))
    sns.countplot(x=sub_df['subject']).set_title(title)
    plt.xlabel("Cфера")
    plt.ylabel("Кількість")
    if savefig:
        plt.savefig('../pics/countplot_{}.png'.format(title))
    else:
        plt.show()
    

def plot_worldcloud(real=True, savefig=True):
    sub_df = df[df['target'] == int(real)]
    stop=set(stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop, background_color="black").generate(str(sub_df['title']))
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if savefig:
        plt.savefig('../pics/worldcloud_{}.png'.format(real))
    else:
        plt.show()


def plot_time_series(real=True, savefig=True):
    sub_df=df[df['target'] == int(real)]
    sub_df=sub_df.groupby(['datetime'])['target'].count()

    plt.figure(figsize=(15,7))
    sns.lineplot(x=sub_df.index, y=sub_df.values)
    if savefig:
        plt.savefig('../pics/time_series_{}.png'.format(real))

    sub_df=pd.DataFrame(sub_df)
    fig = px.line(sub_df)
    fig.show()

