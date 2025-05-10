from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Adjust import path as needed

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Extract form data
        data = CustomData(
            Year=int(request.form.get('Year')),
            Period=request.form.get('Period'),
            Geo_Level=request.form.get('Geo_Level'),
            State=request.form.get('State'),
            Commodity=request.form.get('Commodity'),
            CONDITION=float(request.form.get('CONDITION')),
            PROGRESS=float(request.form.get('PROGRESS')),
            PRICE_RECEIVED=float(request.form.get('PRICE_RECEIVED')),
            STOCKS=float(request.form.get('STOCKS')),
            SALES=float(request.form.get('SALES')),
            Moisture=float(request.form.get('Moisture')),
            sand_per=float(request.form.get('sand_per')),
            slit_per=float(request.form.get('slit_per')),
            clay_per=float(request.form.get('clay_per')),
            ph=float(request.form.get('ph')),
            Cation_Exchange_Capacity=float(request.form.get('Cation_Exchange_Capacity')),
            Organic_Matter=float(request.form.get('Organic_Matter')),
            Available_Water_Capacity=float(request.form.get('Available_Water_Capacity')),
            ksat=float(request.form.get('ksat')),
            slope=float(request.form.get('slope')),
            elev=float(request.form.get('elev')),
            Soil_Texture=request.form.get('Soil_Texture'),
            Soil_Fertility=request.form.get('Soil_Fertility')
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()

        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Render result on the home page
        return render_template('home.html', results=round(results[0], 2))

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)