import torch 
import logging 




def train(model, train_dataloader:torch.utils.data.DataLoader, num_epochs:int, lr:float, device, valid_dataloader:None): 
        model.to(device)
        #Adam with L2 regularization, regularization strength 1e-4
        decay, no_decay = [], []
        for _, param in model.named_parameters():
            if param.ndim == 1:  #bias
                no_decay.append(param)
            else: #weights
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 1e-4},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr
        )

        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        patience = 20
        patience_counter = 0

        logging.info("training started.")
        for epoch in range(num_epochs): 
                model.train()
                train_loss = 0.0
                n_batches = 0
                for X_batch, y_batch in train_dataloader: 
                        # to calculate the mean of loss per batch (train_loss/n_batches)
                        n_batches+=1
                        #tensors should be at the same device as the model
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        #forward pass
                        logits = model(X_batch)
                        batch_loss = criterion(logits, y_batch)

                        train_loss+= batch_loss.item()

                        #backward pass
                        optimizer.zero_grad()
                        batch_loss.backward()

                        #update params 
                        optimizer.step()
                train_loss/= n_batches
                
                if valid_dataloader: 
                    # validation for early stopping
                    val_loss = validate(model, criterion, valid_dataloader, device)

                    logging.debug(
                        f"epoch {epoch:03d} | train cross entropy {train_loss:.4f} | val cross entropy {val_loss:.4f}"
                    )

                    # early stopping
                    if val_loss <= best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        logging.debug(f"patience counter reinitialized.")
                        #save the state of the best model so far, allows as to increase patience, that will prevent stopping because of noise. 
                        torch.save(
                            {"best_model_state":model.state_dict(),
                            "best_validation_loss": best_val_loss,
                            "epoch":epoch}, 
                                    "best_state.pt")
                    else:
                        patience_counter += 1
                        logging.debug(f"validation cross entropy did not improve at epoch {epoch}")

                    if patience_counter >= patience:
                        logging.warning(f"early stopping triggered after {patience} waiting epochs. min cross entropy loss: {best_val_loss}; epoch: {epoch}; best state saved to 'best_state.pt'")
                        break

                if epoch % 10 == 0 or epoch==99: 
                    logging.info(f"epoch {epoch:03d} | train cross entropy {train_loss:.4f} | val cross entropy {val_loss:.4f}")
         
        logging.info("training completed. ")



#do not track gradient
@torch.no_grad()
def validate(model, criterion, valid_dataloader, device):
    "to be used for early stopping inside `train`"

    model.eval()
    total_loss = 0.0
    num_batches = 0
    for X_batch, y_batch in valid_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

           
                  


@torch.no_grad()
def test(trained_model,test_data:torch.utils.data.Dataset, criterion, device):
    "testing using the Recall metric, as we care more about avoiding any "

    trained_model.eval()
    total_loss = 0.0
    num_batches = 0

    dataloader = torch.utils.data.DataLoader(test_data, batch_size=37, shuffle=True,
                                            pin_memory=True) #in case we have GPU
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = trained_model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
                  
