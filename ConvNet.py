# -*- coding: utf-8 -*-
import Imports

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same'),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer
        self.linear = nn.Linear(in_features = 32*7*7, out_features = number_of_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)   
        x = self.linear(x)    
        output = self.softmax(x)
        return output


