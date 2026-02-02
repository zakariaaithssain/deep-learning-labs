import torch
import joblib
import logging

from mlp import MultiLayerPerceptron
from functions import * 
from config import READY_DATA_PATH

from sklearn.model_selection import train_test_split



try:
    data = joblib.load(READY_DATA_PATH)
except FileNotFoundError: 
        print(f"{READY_DATA_PATH} not found.")
        exit(1)

training_data = data["train"]
X_train = training_data["X"]
y_train = training_data["y"]

#we split test data to valid and test, valid wil be used for early stopping. 
X_test, X_valid, y_test, y_valid = train_test_split(data["test"]["X"], data["test"]["y"], test_size=0.5, random_state=123)

#convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32) #float is required for gradient calcs
y_train = torch.tensor(y_train, dtype=torch.long) #long type is issential for cross entropy

X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.long)

num_features = X_train.size(dim=1)
model = MultiLayerPerceptron(input_dim=num_features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=37, shuffle=True,
                                            pin_memory=True) #in case sehel 3lina Allah bshy GPU
valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=37, shuffle=True,
                                        pin_memory=True) 
num_epochs = 500
lr = 1e-3

if __name__ == "__main__": 

    try:
        train(model, train_dataloader, num_epochs, lr, device, valid_dataloader)
        exit(0)
    except KeyboardInterrupt: 
          logging.warning("training interrupted manually.")
          exit(1)
    
    except Exception as e: 
          logging.error(str(e))
          exit(1)