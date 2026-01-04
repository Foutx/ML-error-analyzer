import pandas as pd

class ClassificationErrorAnalyzer:

    def __init__(self, X, y_true, y_pred, preds=None):
        
        self.X = X.reset_index(drop=True)
        self.y_true = y_true.reset_index(drop=True)
        self.y_pred = y_pred
        self.preds = preds

    # ========================== Funcs to analyze =========================================

    def confusion_table(self):

        df = pd.DataFrame({
            "y_true": self.y_true,
            "y_pred": self.y_pred
        })

        return pd.crosstab(df["y_true"], df["y_pred"])
    
    def error_mask(self):

        return self.y_true != self.y_pred
    
    def error_dataframe(self):

        mask = self.error_mask()
        df_errors = self.X[mask].copy()
        df_errors["y_true"] = self.y_true[mask]
        df_errors["y_pred"] = self.y_pred[mask]

        return df_errors
    
    def error_types(self):

        df = self.error_dataframe()

        return (df.groupby(["y_true", "y_pred"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False))
    
    def basic_metrics(self):

        total = len(self.y_true)
        errors = (self.y_true != self.y_pred).sum()

        return {
            "accuracy": 1 - errors / total,
            "error_rate": errors / total,
            "total_samples": total,
            "total_errors": int(errors),
        }
    # =========================================================================================

    # Func to make full analyze
    def analyze(self):

        return {
            "metrics": self.basic_metrics(),
            "confusion": self.confusion_table(),
            "error_types": self.error_types(),
            "n_errors": int((self.y_true != self.y_pred).sum())
        }
