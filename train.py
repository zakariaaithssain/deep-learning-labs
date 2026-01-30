import torch
import joblib
import logging

from config import READY_DATA_PATH



logger = logging.Logger("train")
data = joblib.load(READY_DATA_PATH)

training_data = data["train"]
X = training_data["X"]
y = training_data["y"]
#convert to tensors
X = torch.tensor(X)
y = torch.tensor(y)
#the number of features has increased because of One Hot encoding. 
print(X.shape)
print(y.shape)



def train(model, dataloader, num_epochs:int, lr:float, device) -> list[dict]: 
        #move model to device
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        logging.info("training started.")
        for epoch in range(num_epochs): 
                model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in dataloader: 
                        #tensors should be at the same device as the model
                        X_batch.to(device)
                        y_batch.to(device)

                        #forward pass
                        logits = model(X_batch)
                        batch_loss = criterion(logits, y_batch)

                        epoch_loss+= batch_loss.item()

                        #backward pass
                        optimizer.zero_grad()
                        batch_loss.backward()

                        #update params 
                        optimizer.step()

                logger.debug(f"epoch: {epoch}; cross entropy loss: {epoch_loss:.4f}")

                if epoch % 10 == 0: 
                    logger.info(f"epoch: {epoch}; cross entropy loss: {epoch_loss:.4f}")

        
                        

