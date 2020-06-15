from flask import Flask,render_template,request
import pickle

classifier = pickle.load(open('saved_model.pkl','rb'))
vec = pickle.load(open('vec_transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vec.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction = my_prediction)
    

if __name__ == '__main__':
	app.run(debug=True)
        

