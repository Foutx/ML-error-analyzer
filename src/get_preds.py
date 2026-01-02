import torch

from load_data import DataLoader

class GetPreds:

    def __init__(self, model_path, data_path, target_name):
        
        self.loader = DataLoader()

        self.model_path = model_path
        self.data_path = data_path
        self.target_name = target_name

    # Func to get preds for classification task
    def preds_classif(self):

        try:
            df = self.loader.load_data_classif(self.data_path)

            X_test_df = df.drop(columns=[self.target_name])
            #y_test = df[self.target_name]

            model = self.loader.load_torch_model(self.model_path)

            device = next(model.parameters()).device

            X_test = torch.FloatTensor(X_test_df.values).to(device)

            with torch.no_grad():
                model.eval()
                preds = model(X_test)
                if preds.ndim == 1 or preds.shape[1] == 1:
                    y_pred = (torch.sigmoid(preds) > 0.5).long().squeeze()
                else:
                    y_pred = torch.argmax(preds, dim=1)

            return (
                y_pred.cpu().numpy(),
                preds.cpu().numpy(),
                'binary' if (preds.ndim == 1 or preds.shape[1] == 1) else 'multiclass'
                )
        
        except Exception as e:
            print("Can't get predicts")
            print(e)
            return None