import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self):
        pass

    def load_torch_model(self, model_path):
        
        if not(os.path.exists(model_path)):
            print("No such file or directory!")
            return None
        
        try:
            model = torch.load(model_path, weights_only=True, map_location='cpu')

            if isinstance(model, torch.nn.Module):
                model.eval()
                return model
            
            print("File is not torch model!")
            return None

        except Exception as e:
            print(e)
            return None

    def load_data(self, data_path):
        
        if not(os.path.exists(data_path)):
            print("No such file or directory!")
            return None

        try:
            df = pd.read_csv(data_path)

            return df 

        except Exception as e:
            print(e)
            return None