import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Get absolute paths to model and preprocessor
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            
            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Clean feature names (strip whitespace)
            features.columns = features.columns.str.strip()
            
            # Transform features and make prediction
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        Year: int,
        Period: str,
        Geo_Level: str,
        State: str,
        Commodity: str,
        CONDITION: float,
        PROGRESS: float,
        PRICE_RECEIVED: float,
        STOCKS: float,
        SALES: float,
        Moisture: float,
        sand_per: float,
        slit_per: float,
        clay_per: float,
        ph: float,
        Cation_Exchange_Capacity: float,
        Organic_Matter: float,
        Available_Water_Capacity: float,
        ksat: float,
        slope: float,
        elev: float,
        Soil_Texture: str,
        Soil_Fertility: str):
        
        self.Year = Year
        self.Period = Period
        self.Geo_Level = Geo_Level
        self.State = State
        self.Commodity = Commodity
        self.CONDITION = CONDITION
        self.PROGRESS = PROGRESS
        self.PRICE_RECEIVED = PRICE_RECEIVED
        self.STOCKS = STOCKS
        self.SALES = SALES
        self.Moisture = Moisture
        self.sand_per = sand_per
        self.slit_per = slit_per
        self.clay_per = clay_per
        self.ph = ph
        self.Cation_Exchange_Capacity = Cation_Exchange_Capacity
        self.Organic_Matter = Organic_Matter
        self.Available_Water_Capacity = Available_Water_Capacity
        self.ksat = ksat
        self.slope = slope
        self.elev = elev
        self.Soil_Texture = Soil_Texture
        self.Soil_Fertility = Soil_Fertility

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Year": [self.Year],
                "Period": [self.Period],
                "Geo Level": [self.Geo_Level],
                "State": [self.State],
                "Commodity": [self.Commodity],
                "CONDITION": [self.CONDITION],
                "PROGRESS": [self.PROGRESS],
                "PRICE RECEIVED": [self.PRICE_RECEIVED],
                "STOCKS": [self.STOCKS],
                "SALES": [self.SALES],
                "Moisture": [self.Moisture],
                "sand_per": [self.sand_per],
                "slit_per": [self.slit_per],
                "clay_per": [self.clay_per],
                "ph": [self.ph],
                "Cation Exchange Capacity": [self.Cation_Exchange_Capacity],
                "Organic Matter": [self.Organic_Matter],
                "Available Water Capacity": [self.Available_Water_Capacity],
                "ksat": [self.ksat],
                "slope": [self.slope],
                "elev": [self.elev],
                "Soil_Texture": [self.Soil_Texture],
                "Soil_Fertility": [self.Soil_Fertility]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)