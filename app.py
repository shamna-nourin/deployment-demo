from flask import Flask, render_template, request
import pickle
import numpy as np

# from sklearn import preprocessing
# label_encoder=preprocessing.LabelEncoder()

app = Flask(__name__)
# Load the trained model    

model=pickle.load(open('model.pkl','rb'))
print('model loaded')

Fuel_Type_en=pickle.load(open('Fuel_Type.pkl','rb'))
Transmission_en=pickle.load(open('Transmission.pkl','rb'))
#scaled = pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

# @app.route('/predict', methods=['POST'])
# def predict():
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        try:
            # Get form inputs
            Year = int(request.form['Year'])
            print(Year)
            Present_Price = float(request.form['Present_Price'])
            print(Present_Price)
            Kms_Driven = float(request.form['Kms_Driven'])
            print(Kms_Driven)
            Fuel_Type = int(request.form['Fuel_Type'])
            print(Fuel_Type)
            Transmission = request.form['Transmission']
            print(Transmission)
            Owner = int(request.form['Owner'])
            print(Owner)
            Seller_Type_Individual = int(request.form['Seller_Type_Individual'])
            print(Seller_Type_Individual)

            # Fuel_Type_val = Fuel_Type_en.transform([Fuel_Type])[0]
            # print(Fuel_Type_val)

            Transmission_val = Transmission_en.transform([Transmission])[0]
            print(Transmission_val)

            # Prepare data 
            details =[Year, Present_Price, Kms_Driven, Fuel_Type, int(Transmission_val),Owner,Seller_Type_Individual]
            print(details)

            data_out=np.array(details).reshape(1,-1)
            print(data_out)
            print(data_out.shape)

            scaled = pickle.load(open('scaling.pkl','rb'))
            data_scaled = scaled.transform(data_out)


            # data_scaled = scaled.transform(data_out)
            # print("Scaled input:", data_scaled)
            
            # Predict car price
            prediction = model.predict(data_scaled)
            #prediction = model.predict(input_features)[0]
            print(prediction)

            return render_template('index.html', prediction_text=f'Estimated Car Price: {float(round(prediction[0],2))}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
