from flask import Flask
from flask.wrappers import Request
from requests.models import Response
from flask import request
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle

app = Flask(__name__)

def process_msg(msg):
    if msg == "hi":
        response = "Hello, Welcome to the cyberbullying detection bot!"
    else:
        
        msg = [msg]
        
     
        my_file = open("C:/Users/mishr/Documents/FinalProject/FinalProject/service_testing/stopwords.txt", "r")
        content = my_file.read()
        content_list = content.split("\n")
        my_file.close()

        tfidf_vector = TfidfVectorizer(stop_words = content_list, lowercase = True,vocabulary=pickle.load(open("C:/Users/mishr/Documents/FinalProject/FinalProject/service_testing/tfidf_vector_vocabulary.pkl", "rb")))
        data=tfidf_vector.fit_transform(msg)
        print(data)
        model = pickle.load(open("C:/Users/mishr/Documents/FinalProject/FinalProject/service_testing/LinearSVC.pkl", 'rb'))
        pred = model.predict(data)
        response = str(pred[0])
        print(response)
        if(response=='1'):
            response = "Output from my ML Model :- bullying"
        else:
            response = "Output from my ML Model :- non-bullying"
            

    return response 




@app.route("/testing", methods = ["POST"])
def testing():
    f=request.form 
    print(f['Body'])
    msg=f['Body']
    sender=f['From']
    print(msg)
    response = process_msg(msg)
    return response,200