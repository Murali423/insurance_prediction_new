from insurance.pipeline.training_pipeline import start_training_pipeline

file_path = '/config/workspace/insurance.csv'

if __name__ == "__main__":
    try:
        start_training_pipeline()
    except Exception as e:
        raise e