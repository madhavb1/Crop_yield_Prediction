from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Make sure this matches your form's action URL
@app.route('/predict', methods=['GET', 'POST'])  # Changed to just '/predict'
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
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
        
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=f"Predicted Yield: {results[0]:.2f}")
    
    except Exception as e:
        return render_template('home.html', error_message=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)