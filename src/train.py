import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from src.load_dataset import load_df, split_dataset
from src.preprocessing import preprocess_text, run_preprocess, create_features
from src.config import DatasetConfig

class MessageClModel:
    def __init__(self, dictionary, le):
        self.model = GaussianNB()
        self.dictionary = dictionary
        self.le = le
        
    def train(self, x_train, y_train):
        print('Start training...')
        self.model.fit(x_train, y_train)
        print('Training completed!')
        
    def evaluate(self, x_val, y_val):
        y_val_pred = self.model.predict(x_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        return val_accuracy
    
    def predict(self, text):
        tokens = preprocess_text(text)
        features = create_features(tokens, self.dictionary)
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        prediction_cls = self.le.inverse_transform(prediction)[0]
        
        # Map "ham" and "spam" to "Not Spam" and "Spam"
        if prediction_cls == 'spam':
            return "Spam"
        else:
            return "Not Spam"

def main():
    df = load_df(DatasetConfig.DATASET_PATH)
    x, y, dictionary, le = run_preprocess(df)
    x_train, y_train, _, _, _, _ = split_dataset(x, y)
    model = MessageClModel(dictionary=dictionary,
                           le=le)
    model.train(x_train=x_train,
                y_train=y_train)
    
    return model

model = main()