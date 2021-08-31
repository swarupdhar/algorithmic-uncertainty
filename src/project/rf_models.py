import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from matplotlib import pyplot as plt

class BasicRFModel:
    def __init__(self, X:pd.DataFrame, Y:pd.Series, test_size=0.2, random_state=1) -> None:
        self.X = X
        self.Y = Y
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X,
            Y,
            test_size=test_size,
            random_state=random_state
        )
        self.model = RandomForestClassifier()
    
    def fit(self) -> None:
        self.model.fit(self.X_train, self.Y_train)
    
    def predict(self, x) -> numpy.ndarray:
        return self.model.predict(x)
    
    def get_accuracy(self):
        return accuracy_score(self.Y_test, self.predict(self.X_test))
    
    def plot_roc(self, label=None):
        scores = self.model.predict_proba(self.X_test)
        fpr, tpr, _ = roc_curve(self.Y_test, scores[:, 1])
        auc_score = roc_auc_score(self.Y_test, scores[:, 1])
        
        plt.plot(
            fpr,
            tpr,
            marker='.',
            label=f"{label if label else ''} (AUC={auc_score:.2f})"
        )
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()