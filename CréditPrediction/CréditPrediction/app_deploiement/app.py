import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Faire la prédiction
    prediction = model.predict(final_features)
    output = int(prediction[0])  # 0 ou 1 en général

    # Rediriger selon la prédiction
    if output == 1:
        return render_template('success.html')
    else:
        return render_template('rejected.html')


if __name__ == "__main__":
    app.run(debug=True)