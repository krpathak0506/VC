from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df1 = pd.read_csv("dataset/Youtube01-Psy.csv")              
    df2 = pd.read_csv("dataset/Youtube02-KatyPerry.csv")        
    df3 = pd.read_csv("dataset/Youtube03-LMFAO.csv")            
    df4 = pd.read_csv("dataset/Youtube04-Eminem.csv")           
    df5 = pd.read_csv("dataset/Youtube05-Shakira.csv")          

    # Merge all the datasset into single file
    frames = [df1,df2,df3,df4,df5]                          
    df_merged = pd.concat(frames)                           
    keys = ["Psy","KatyPerry","LMFAO","Eminem","Shakira"]   
    df_with_keys = pd.concat(frames,keys=keys)              
    dataset=df_with_keys

    # working with text content
    dataset = dataset[["CONTENT" , "CLASS"]]            

    # Predictor and Target attribute
    dataset_X = dataset['CONTENT']                       
    dataset_y = dataset['CLASS']                         

    # Extract Feature With TF-IDF model 
    corpus = dataset_X                               
    cv = TfidfVectorizer()                          
    X = cv.fit_transform(corpus).toarray()          


    # import pickle file of my model
    model = open("model/model.pkl","rb")
    clf = pickle.load(model)
    
    if request.method == 'POST':
    	comment = request.form['comment']
    	data = [comment]
    	vect = cv.transform(data).toarray()
    	my_prediction = clf.predict(vect)
    	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)