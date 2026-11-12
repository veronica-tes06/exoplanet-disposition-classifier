import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cumulative.csv')
LABEL = 'koi_disposition'
ID_COLUMN = 'kepoi_name'

# The six values you can type into the GUI.
CORE_FEATURES = ['koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']

# Measurements from the transit fit and the signal itself.
SIGNAL_FEATURES = [
    'koi_model_snr', 'koi_num_transits', 'koi_max_mult_ev', 'koi_max_sngle_ev', 'koi_bin_oedp_sig',
    'koi_ror', 'koi_srho', 'koi_dor', 'koi_incl', 'koi_sma', 'koi_insol',
    'koi_period_err1', 'koi_duration_err1', 'koi_depth_err1', 'koi_impact_err1',
    'koi_prad_err1', 'koi_ror_err1', 'koi_srho_err1', 'koi_dor_err1',
]

# Properties of the host star.
STAR_FEATURES = ['koi_steff', 'koi_slogg', 'koi_smet', 'koi_srad', 'koi_smass', 'koi_kepmag']

# Centroid shifts: does the star's light move during transit? A big shift means the
# dip is probably coming from a different, background star.
CENTROID_FEATURES = [
    'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sra_err', 'koi_fwm_sdec', 'koi_fwm_sdec_err',
    'koi_fwm_srao', 'koi_fwm_srao_err', 'koi_fwm_sdeco', 'koi_fwm_sdeco_err',
    'koi_fwm_prao', 'koi_fwm_prao_err', 'koi_fwm_pdeco', 'koi_fwm_pdeco_err',
    'koi_dicco_mra', 'koi_dicco_mra_err', 'koi_dicco_mdec', 'koi_dicco_mdec_err',
    'koi_dicco_msky', 'koi_dicco_msky_err', 'koi_dikco_mra', 'koi_dikco_mra_err',
    'koi_dikco_mdec', 'koi_dikco_mdec_err', 'koi_dikco_msky', 'koi_dikco_msky_err',
]

# How many other signals were found around the same star.
SYSTEM_FEATURES = ['koi_count', 'koi_tce_plnt_num']

ALL_FEATURES = CORE_FEATURES + SIGNAL_FEATURES + STAR_FEATURES + CENTROID_FEATURES + SYSTEM_FEATURES

# Deliberately NOT used, because they give the answer away:
#   koi_pdisposition, koi_score, koi_fpflag_*  - outputs of NASA's own vetting
#   kepler_name                                - only confirmed planets get a Kepler-xx name
#   koi_vet_*, koi_disp_prov, koi_comment      - bookkeeping about the label itself

MODEL_TYPES = ['Gradient Boosting', 'Decision Tree']


def add_engineered_features(X):
    X = X.copy()
    # Transit duration compared with what Kepler's third law predicts for that period.
    X['dur_norm'] = X['koi_duration'] / np.cbrt(X['koi_period'])
    # Impact parameter >= 1 means the planet only grazes the edge of the star.
    X['grazing'] = (X['koi_impact'] >= 1).astype(float)
    return X


def build_model(model_type, seed):
    if model_type == 'Decision Tree':
        # Decision trees can't handle missing values, so fill them with the median first.
        return make_pipeline(
            SimpleImputer(strategy='median'),
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=seed))
    # Gradient boosting handles missing values on its own.
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, early_stopping=False, random_state=seed)


class MyDecisionTree:

    def __init__(self):
        self.df = pd.read_csv(DATA_PATH, comment='#', low_memory=False)
        self.df = self.df.dropna(subset=[LABEL]).reset_index(drop=True)
        self.y = self.df[LABEL]
        self.seedValue = 1
        self.modelType = MODEL_TYPES[0]
        self.train_idx = None
        self.test_idx = None
        self.tree = None        # full model, uses every feature
        self.coreTree = None    # model that only uses the six values typed into the GUI
        self.y_hat = None

    def isTrained(self):
        return self.tree is not None

    def updateSeed(self, newValue):
        self.seedValue = newValue

    def updateModelType(self, newType):
        self.modelType = newType

    def features(self, feature_list=ALL_FEATURES, rows=None):
        X = self.df[feature_list] if rows is None else self.df.loc[rows, feature_list]
        return add_engineered_features(X)

    def trainData(self):
        self.train_idx, self.test_idx = train_test_split(
            self.df.index, test_size=0.2, random_state=self.seedValue, stratify=self.y)
        y_train = self.y[self.train_idx]

        self.tree = build_model(self.modelType, self.seedValue)
        self.tree.fit(self.features(rows=self.train_idx), y_train)

        self.coreTree = build_model(self.modelType, self.seedValue)
        self.coreTree.fit(self.features(CORE_FEATURES, self.train_idx), y_train)

    def testData(self):
        y_test = self.y[self.test_idx]
        self.y_hat = self.tree.predict(self.features(rows=self.test_idx))
        cm = confusion_matrix(y_test, self.y_hat, labels=self.categories())
        return cm, accuracy_score(y_test, self.y_hat), f1_score(y_test, self.y_hat, average='macro')

    def testSize(self):
        return len(self.test_idx)

    def makePrediction(self, period, impact, duration, depth, prad, teq):
        """Predict from the six typed-in values, using the 6-feature model."""
        input_data = pd.DataFrame([[period, impact, duration, depth, prad, teq]], columns=CORE_FEATURES)
        X = add_engineered_features(input_data)
        return self.coreTree.predict(X)[0], self._probabilities(self.coreTree, X)

    def predictKoi(self, koi_name):
        """Predict a Kepler object from the dataset, using every feature.

        Returns None if the name isn't found, otherwise a dict with the prediction,
        the class probabilities, the real label, the six core values, and whether
        the object was in the training set (in which case the prediction is not a fair test).
        """
        matches = self.df.index[self.df[ID_COLUMN].str.upper() == koi_name.strip().upper()]
        if len(matches) == 0:
            return None
        row = matches[0]
        X = self.features(rows=[row])
        return {
            'prediction': self.tree.predict(X)[0],
            'probabilities': self._probabilities(self.tree, X),
            'actual': self.y[row],
            'core_values': self.df.loc[row, CORE_FEATURES].tolist(),
            'in_training_set': row in set(self.train_idx),
        }

    def exampleTestKois(self, n=3):
        """A few KOI names from the test set, one per class where possible."""
        test = self.df.loc[self.test_idx]
        return [test[test[LABEL] == c][ID_COLUMN].iloc[0] for c in self.categories()][:n]

    @staticmethod
    def _probabilities(model, X):
        return dict(zip(model.classes_, model.predict_proba(X)[0]))

    def categories(self):
        return sorted(set(self.y))

    def readCategories(self):
        return ' / '.join(self.categories())
