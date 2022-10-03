import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import speech_recognition as sr
import pyttsx3
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import wikipedia

r=sr.Recognizer()
engine = pyttsx3.init()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# End of training model

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def listen():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,duration=1)
        print("\nSay something : ")
        print('You -> ', end=' ')
        audio= r.listen(source)
        try:
            text = r.recognize_google(audio)
            rans = text
        except:
            rans = "Sorry, could not recognise. Please try again"
    return rans

def chat():
    print("\t\t\t      =================")
    print("============================> ∥ Bot Activated ∥ <==================================")
    print("\t\t\t      =================\n")
    print('===================================================================================')
    print('∥\t\t\t\tInstructions\t\t\t\t\t  ∥')
    print("∥\t\t\t\t\t\t\t\t\t\t  ∥")
    print('∥\t-> To provide input you can use text aur audio.\t\t\t\t  ∥')
    print("∥\t-> You can switch the input method by typing 'switch' as a command.\t  ∥")
    print("∥\t-> You can add 'google' in command to google stuff.\t\t\t  ∥")
    print("∥\t-> You can type or say 'turn off' to deacivate the bot.\t\t\t  ∥")
    print('===================================================================================')

    print('\nHow would you like to Communicate with the bot?\nType 1 for text\nType 2 for audio')
    print('--->', end=' ')
    input_way = int(input())

    while True:
        if input_way == 1:
            print('\nType somthing : ')
            inp = input("You -> ")
            if inp.lower() == "turn off":
                print("\t\t\t      ====================")
                print("============================> ∥  Bot Deactivated ∥ <==================================")
                print("\t\t\t      ====================")
                break
        elif input_way == 2:
            user_input = listen()
            inp = user_input
            print(user_input)
            if inp.lower() == "turn off":
                print("\t\t\t      ====================")
                print("============================> ∥  Bot Deactivated ∥ <==================================")
                print("\t\t\t      ====================")
                break
        
        if 'google' in inp.lower():
                filtered_inp = inp.replace('google', '')
                info = wikipedia.summary(filtered_inp, sentences=3)
                print('Bot : ', info)
                engine.say(info)
                engine.runAndWait()
                continue

        if(inp == 'switch'):
            if(input_way == 1):
                input_way = 2
                print('\n==>Switched to audio input!')
                continue
            elif(input_way == 2):
                input_way = 1
                print('\n==> Switched to text input!')
                continue
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            ans = random.choice(responses)
            print('Bot -> ', ans)
            engine.say(ans)
            engine.runAndWait()
        else:
            print("Bot -> I didn't get that, try again.")
            engine.say("I didn't get that, try again.")
            engine.runAndWait()

chat()