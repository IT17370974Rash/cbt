from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import re
import os
import tensorflow as tf
from numpy import array
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from doc3 import training_doc3
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tensorflow.keras.utils import to_categorical
from actvity import act_list

app = Flask(__name__)

cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
tokens = word_tokenize(cleaned)
train_len = 4+1
text_sequences = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)
    
sequences = {}
count = 1

for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)
vocabulary_size = len(tokenizer.word_counts)+1

def init():
    global model,graph
    # load the pre-trained Keras model
    MODEL_PATH ='project\cbtmodel.h5'
    model = load_model(MODEL_PATH)
    # graph = tf.get_default_graph()
    graph = tf.compat.v1.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        word = request.form.get('word')

        activity_list = []
        input_text = word.strip().lower()
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=4, truncating='pre')
        #print(encoded_text, pad_encoded)
        for i in (model.predict(pad_encoded)[0]).argsort()[-6:][::-1]:
            # pred_word = tokenizer.index_word[i]
            pred_word = act_list[i+1]
            activity_list.append(pred_word)
            ##print("Next activity suggestion:",pred_word)

        return render_template('result.html',
        aa=activity_list[0],
        ab=activity_list[1],
        ac=activity_list[2],
        ad=activity_list[3],
        ae=activity_list[4],
        af=activity_list[5])


    return render_template("activites.html")
    
if __name__ == "__main__":
    init()
    app.run()