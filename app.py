from flask import Flask, render_template, request, redirect
import speech_recognition as sr     # Speech Recognizer
from nltk import tokenize       # Natural Language Toolkit used to extract keywords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from operator import itemgetter  # itemgetter will help sort dictionary
import math
stop_words = set(stopwords.words('english'))
app = Flask(__name__)


def totalWords(text):   #returns a list of words in text
    words = text.split()
    return words


def totalSentences(text):   #returns list of sentences in text
    sentences = tokenize.sent_tokenize(text)
    return sentences


def wordFreq(words):    #calculate the Term Frequency(TF) for each word in text
    freq = {}   #dictionary to store each word and its corresponding TF score
    for word in words:
        word = word.replace(".", "")
        if word not in stop_words:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    words_size = len(words)
    #Divide each element in 'freq' dictionary by the total number of words
    freq.update((x, y/int(words_size)) for x, y in freq.items())
    return freq


def check_sent(word, sentences):    #checks if a word is present in a sentence list
    #check if word is present in a sentence
    final = [all([w in x for w in word]) for x in sentences]
    #stores word in sentence if word appears
    sent_length = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_length))


def calcIDF(words, sentences):  #calculate Inverse Document Frequency(IDF)
    idf = {}
    for word in words:
        word = word.replace(".", "")
        if word not in stop_words:
            if word in idf:
                idf[word] = check_sent(word, sentences)
            else:
                idf[word] = 1
    sent_len = len(sentences)
    #Perform necessary IDF equation to get score
    idf.update((x,math.log(int(sent_len)/y)) for x, y in idf.items())
    return idf


def calcScore(tf, idf): #calculate TF*IDF
    return {key: tf[key]*idf.get(key,0) for key in tf.keys()}


def keyWords(words, n): #return 'n' most important keywords from 'words'
    result = dict(sorted(words.items(), key = itemgetter(1), reverse = True)[:n])
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    keywords = ""
    if request.method == "POST":    #file is received
        if "file" not in request.files:     #error-checking
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)
            words = totalWords(transcript)
            sentences = totalSentences(transcript)
            tf = wordFreq(words)
            idf = calcIDF(words,sentences)
            score = calcScore(tf, idf)
            keywords = keyWords(score, 10)
    return render_template("index.html", transcript=transcript, keywords=keywords)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)