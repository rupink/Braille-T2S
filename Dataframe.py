# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:30:56 2023

@author: rupin
"""
import Imports

class Dataframe():
    def __init__(self, df):

        for index, row in df.iterrows():
          try:
            # img = Image.open("gdrive/MyDrive/ECE324_Project/data/character_data/" + row.filename)
            img = Image.open("character_data/" + row.filename)
            if index < 566:
              df.at[index, "Image Object"] = img
            if index >= 566:
              try:
                changed = transforms.RandAugment()(img)
                df.at[index, "Image Object"] = changed
              except OSError:
                pass
          except FileNotFoundError:
            print("not found")
            pass
        
        for index, row in df.iterrows():
          try:
            # img = Image.open("gdrive/MyDrive/ECE324_Project/data/character_data/" + row.filename)
            img = Image.open("character_data/" + row.filename)
            df.at[index, "Image Object"] = img
          except FileNotFoundError:
            pass
        
        self.df = df.dropna(subset = ['Image Object']).reset_index(drop=True)
        
    def encoder(self):
        encoder = OneHotEncoder()
        encoder_df = pd.DataFrame(encoder.fit_transform(self.df[['label']]).toarray())
        encoder_df.columns = encoder.get_feature_names_out(['label'])
        self.final_df = df.join(encoder_df)
    
    def split_data(self):
        X = self.final_df['Image Object']
        #y = final_df['label']
        y = self.final_df[final_df.columns[5:-1]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.y_train =  X_train.reset_index(drop=True), y_train.reset_index(drop=True)
        