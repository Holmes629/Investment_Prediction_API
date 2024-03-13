from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app= Flask(__name__)
# Load the trained model from the .pkl file
with open('ml_models_package.pkl', 'rb') as f:
    model = pickle.load(f)
    
@app.route('/', methods= ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features= [int(x) for x in request.form.values()]
    final_features= [np.array(int_features)]
    prediction1= model['Linear Regression'].predict(final_features)
    prediction2= model['Lasso Regression'].predict(final_features)
    prediction3= model['Polynomial Regression'].predict(final_features)
    prediction4= model['Ridge Regression'].predict(final_features)
    
    output1= round(prediction1[0], 2)
    output2= round(prediction2[0], 2)
    output3= round(prediction3[0], 2)
    output4= round(prediction4[0], 2)
    
    return render_template('index.html', prediction_text= 'Predicted value using Linear Regression is: $ {},\nPredicted value using Lasso Regression is: ${},\nPredicted value using Polynomial Regression is: $ {},\nPredicted value using Ridge Regression is: $ {}'.format(output1, output2, output3, output4))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data= request.get_json(force=True)
    prediction= model['Linear Regression'].predict([np.array(list(data.values()))])
    
    output= prediction[0]
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(port= 3000, debug=True)
    # print(model['Linear Regression'].predict(np.array([[1, 1, 1, 1, 0]])))
    