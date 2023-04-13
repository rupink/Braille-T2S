# -*- coding: utf-8 -*-
import Imports
class MyDataset(torch.utils.data.Dataset):

  def __init__(self, X, y, transform=None):
    self.X = X
    self.y = y
    self.transform = transform

  def __len__(self):
    return len(self.y)
  
  def __getitem__(self, i):
    # return self.x[idx],self.y[idx]
    label = self.y[i]
    image = self.X[i]
    if self.transform:
        image = self.transform(image)
    sample = {"Image": image, "Label": label}
    return sample

