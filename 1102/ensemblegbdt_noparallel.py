from numpy import np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class EnsembleGBDT (object):
    def __init__(self, n_gbdts=1, sample_size=0.1):
        self.n_gbdts = n_gbdts
        self.yyhgbdt = []
        self.sample_size = sample_size

    def __repr__(self):
        return "Number of GBDT: {0}, Sample size: {1}.".format (self.n_gbdts, self.sample_size)

    def fit(self, train_set_x, train_set_y):
        self.yyhgbdt = [gbdt.fit (XY[:][0], XY[:][2]) for (XY, gbdt) in [(train_test_split (train_set_x, train_set_y,
                                                                                            test_size=1 - self.sample_size,
                                                                                            random_state=i),
                                                                          GradientBoostingClassifier (learning_rate=0.1,
                                                                                                      n_estimators=100,
                                                                                                      subsample=1,
                                                                                                      min_samples_split=20,
                                                                                                      min_samples_leaf=10,
                                                                                                      max_depth=5,
                                                                                                      max_features=None,
                                                                                                      verbose=0,
                                                                                                      max_leaf_nodes=None,
                                                                                                      warm_start=False,
                                                                                                      random_state=i))
                                                                         for i in range (self.n_gbdts)]]

    def predict_proba(self, test_set_x):
        prediction = [model.predict_proba (test_set_x)[:, 1] for model in self.yyhgbdt]
        prediction_test = np.mean (np.array (prediction), axis=0)
        return prediction_test
