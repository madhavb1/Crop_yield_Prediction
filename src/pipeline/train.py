from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    # Provide paths to your train and test data
    train_path = "E:/Crop_yield_Prediction/notebook/train.txt"
    test_path = "E:/Crop_yield_Prediction/notebook/test.txt"
    
    # Initialize and run training pipeline
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline(train_path, test_path)
    
    print("Training completed successfully!")