import os

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exoplanet.csv')
FEATURES = ['period', 'impact', 'duration', 'depth', 'prad', 'teq']


class MyDecisionTree:

    def __init__(self):
        self.df = pd.read_csv(DATA_PATH, comment='#').dropna()
        self.X = self.df[FEATURES]
        self.y = self.df.disposition
        self.seedValue = 1
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tree = None
        self.y_hat = None

    def isTrained(self):
        return self.tree is not None

    def updateSeed(self, newValue):
        self.seedValue = newValue

    def trainData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=250, random_state=self.seedValue, stratify=self.y)

        self.tree = DecisionTreeClassifier(random_state=self.seedValue)
        self.tree.fit(self.X_train, self.y_train)

    def makePrediction(self, period, impact, duration, depth, prad, teq):
        input_data = pd.DataFrame([[period, impact, duration, depth, prad, teq]], columns=FEATURES)
        return self.tree.predict(input_data)[0]

    def testData(self):
        self.y_hat = self.tree.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_hat, labels=self.categories())
        return cm, accuracy_score(self.y_test, self.y_hat)

    def categories(self):
        return sorted(set(self.y))

    def readCategories(self):
        return ' / '.join(self.categories())
