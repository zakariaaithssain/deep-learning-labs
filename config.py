import logging


READY_DATA_PATH = "data/ready/ready.joblib"
LOG_LEVEL = logging.DEBUG




logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler("train.log", mode="w"), 
        logging.StreamHandler()
    ], 
    level=LOG_LEVEL
)