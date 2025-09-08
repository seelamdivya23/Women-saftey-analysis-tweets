import tkinter
from textblob import TextBlob
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import nltk

# Download stopwords if not present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

main = tkinter.Tk()
main.title("Analysis of Women Safety in Indian Cities Using Machine Learning on Tweets")
main.geometry("1300x1200")

global filename
tweets_list = []
clean_list = []
global pos, neu, neg

def tweetCleaning(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    if len(tokens) == 0:
        return "[EMPTY AFTER CLEANING]"
    return ' '.join(tokens)

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def read():
    text.delete('1.0', END)
    tweets_list.clear()
    try:
        train = pd.read_csv(filename, encoding='iso-8859-1')

        # Handle column case
        if 'Text' in train.columns:
            col = 'Text'
        elif 'text' in train.columns:
            col = 'text'
        else:
            text.insert(END, "Error: Dataset must have a column named 'Text'\n")
            return

        for tweet in train[col]:
            tweets_list.append(str(tweet))
            text.insert(END, str(tweet) + "\n")

        text.insert(END, "\n\nTotal tweets found in dataset: " + str(len(tweets_list)) + "\n\n\n")

    except Exception as e:
        text.insert(END, f"Error while reading dataset: {e}\n")

def clean():
    text.delete('1.0', END)
    clean_list.clear()

    if len(tweets_list) == 0:
        text.insert(END, "Error: Please read dataset first!\n")
        return

    for tweet in tweets_list:
        original = tweet.strip()
        cleaned = tweetCleaning(original.lower())
        clean_list.append(cleaned)
        text.insert(END, f"Original: {original}\nCleaned: {cleaned}\n\n")

    text.insert(END, "\n\nTotal tweets cleaned: " + str(len(clean_list)) + "\n\n\n")

def machineLearning():
    text.delete('1.0', END)
    global pos, neu, neg
    pos, neu, neg = 0, 0, 0
    for tweet in clean_list:
        blob = TextBlob(tweet)
        if blob.polarity <= 0.2:
            neg += 1
            sentiment = "NEGATIVE"
        elif blob.polarity <= 0.5:
            neu += 1
            sentiment = "NEUTRAL"
        else:
            pos += 1
            sentiment = "POSITIVE"

        text.insert(END, f"{tweet}\nPredicted Sentiment : {sentiment}\nPolarity Score      : {blob.polarity}\n")
        text.insert(END, '====================================================================================\n')

def graph():
    label_X = ['Positive', 'Negative', 'Neutral']
    category_X = [pos, neg, neu]
    text.delete('1.0', END)
    text.insert(END, "Safety Factor\n\n")
    text.insert(END, f'Positive : {pos}\n')
    text.insert(END, f'Negative : {neg}\n')
    text.insert(END, f'Neutral  : {neu}\n\n')
    text.insert(END, f'Length of tweets  : {len(clean_list)}\n')
    if len(clean_list) > 0:
        text.insert(END, f'Positive : {pos} / {len(clean_list)} = {pos/len(clean_list):.2f}\n')
        text.insert(END, f'Negative : {neg} / {len(clean_list)} = {neg/len(clean_list):.2f}\n')
        text.insert(END, f'Neutral  : {neu} / {len(clean_list)} = {neu/len(clean_list):.2f}\n')

    plt.pie(category_X, labels=label_X, autopct='%1.1f%%')
    plt.title('Women Safety & Sentiment Graph')
    plt.axis('equal')
    plt.show()

# UI Setup
font = ('times', 16, 'bold')
title = Label(main, text='Analysis of Women Safety in Indian Cities Using Machine Learning on Tweets')
title.config(bg='brown', fg='white', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Tweets Dataset", command=upload, font=font1)
uploadButton.place(x=50, y=100)

pathlabel = Label(main, bg='brown', fg='white', font=font1)
pathlabel.place(x=370, y=100)

readButton = Button(main, text="Read Tweets", command=read, font=font1)
readButton.place(x=50, y=150)

cleanButton = Button(main, text="Tweets Cleaning", command=clean, font=font1)
cleanButton.place(x=210, y=150)

mlButton = Button(main, text="Run Machine Learning Algorithm", command=machineLearning, font=font1)
mlButton.place(x=400, y=150)

graphButton = Button(main, text="Women Safety Graph", command=graph, font=font1)
graphButton.place(x=730, y=150)

font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=150, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=200)

main.config(bg='brown')
main.mainloop()
