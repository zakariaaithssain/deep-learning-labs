from src.functions import test, report_and_conf_matrix
from src.mlp import MultiLayerPerceptron
from src.config import X_test, y_test, DOTPT_FILE_PATH

import torch
import logging


if __name__ == "__main__":  
    logger = logging.getLogger("test")

    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = X_test.size(dim=-1)
    model = MultiLayerPerceptron(input_dim)
    checkpoint = torch.load(DOTPT_FILE_PATH, map_location=device)
    model.load_state_dict(checkpoint["best_model_state"])

    try:
        #this calculates only recall and sends it to the test.log file
        test(model, test_data=test_data, device=device)
        #this gives a complete classification report to terminal, and plots a confusion matrix
        report_and_conf_matrix(model, X_test, y_test)
        exit(0)
    except KeyboardInterrupt: 
          logger.warning("testing interrupted manually.")
          exit(1)
    
    except Exception as e: 
          logger.error(str(e))
          raise
