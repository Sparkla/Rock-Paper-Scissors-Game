from flask import Flask, render_template, request
import numpy as np
import tensorflow.python.keras.models
from tensorflow.python.keras.preprocessing import image
import re
import sys 
import os
import base64
sys.path.append(os.path.abspath("./model"))
from load import * 


global graph, model

model, graph = init()

app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
	    output.write(base64.b64decode(imgstr))

@app.route('/next/',methods=['GET','POST'])
def next():
	return render_template('index.html')



@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)
	x = image.load_img('output.png',target_size = (64,64))
	x = image.img_to_array(x)
	x = np.expand_dims(x,axis=0)
	
	#with graph.as_default():
	result = model.predict(x)
	print(result)
	response = 'YO'
	if result[0][0]==1:
		print("paper")
		response = 'PAPER'
	elif result[0][1]==1:
		print('rock')
		response = 'ROCK'
	elif result[0][2]==1:
		print('scissors')
		response = 'SCISSORS'

	return response	

if __name__ == '__main__':
    app.run(debug=True, port=8000)