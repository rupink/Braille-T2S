# -*- coding: utf-8 -*-
from PIL import ImageFile
import Imports

def one_hot_to_ind(b_y):
  ind = torch.empty(len(b_y))
  #print(b_y)\
  #print(torch.argsort(b_y))
  for i in range(len(b_y)):
    ind[i] = torch.argsort(b_y[i])[-1]
  return ind    

def Trainer(epochs,batch,optimizer,criterion,learning_rate,model,train_loader):

    avg_train_loss_epoch = []
    avg_val_loss_epoch = []
    train_acc_epoch = []
    val_acc_epoch = []
    
    for epoch in range(epochs):
      avg_train_loss = 0 # (averaged) training loss per batch
      avg_val_loss =  0  # (averaged) validation loss per batch
      train_acc = 0      # training accuracy per batch
      val_acc = 0    
        
      for samples in train_loader:
        b_x = samples['Image']
    
        #print(samples["Label"])
    
        b_y = samples['Label']
        y_pred = model(b_x)
        y_pred_ind = one_hot_to_ind(y_pred)
        y_pred_ind = y_pred_ind.type(torch.long)
    
        b_y = one_hot_to_ind(b_y)
        b_y = b_y.type(torch.long)
    
        #print(torch.argsort(y_pred[0])[-1])
        #print(y_pred_ind)
    
    
    
        loss = criterion(y_pred,b_y)
    
        avg_train_loss +=loss
        train_acc += (b_y == y_pred_ind).sum()
    
        optimizer.zero_grad()
        loss.backward(retain_graph=True) 
        optimizer.step()
    
        avg_train_loss_epoch.append(avg_train_loss/batch)
        train_acc_epoch.append(train_acc / batch)
    
      if epoch % 10 == 0:
        print("Loss = ", loss.item(), "Epoch = ", epoch, "avg_train_loss = ", avg_train_loss.item()/batch, "train_acc = ", train_acc.item()/(batch*batch))  
    
    return avg_train_loss_epoch, avg_val_loss_epoch, train_acc_epoch, val_acc_epoch