import logging


READY_DATA_PATH = "data/ready/ready.joblib"

logging.basicConfig(
    handlers=[
        logging.FileHandler("train.log"), 
        logging.StreamHandler()
    ]
)