from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from scipy.stats.distributions import chi2
import numpy as np

def from_pred_prob_to_pred_class(y_pred_prob):
    return np.argmax(y_pred_prob, axis=1)

class StatTest:
    """
        y_true: (n_samples,)
        y_prediction: (n_samples,) - predicted classes
        y_probabilities: (n_samples, n_classes) - probabilities of the classes (summing to 1)
    """
    def eval(self, y_true, y_prediction, y_probabilities):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_results(self):
        raise NotImplementedError("Has be implemented in the sub class")


class McNemarTest(StatTest):
    def __init__(self):
        self.chi2_stats = []
        self.p_values = []

    def eval(self, y_pred_prob_1, y_pred_prob_2)-> np.array:
        y_true_1 = y_pred_prob_1[:,0]
        y_true_2 = y_pred_prob_2[:,0]
        y_pred_1 = from_pred_prob_to_pred_class(y_pred_prob_1[:,1:])
        y_pred_2 = from_pred_prob_to_pred_class(y_pred_prob_2[:,1:])

        if not  np.array_equal(y_true_1,y_true_2):
            raise ValueError("True labels are not the same")
        
        y_pred_match_1 = np.equal(y_pred_1, y_true_1)
        y_pred_match_2 = np.equal(y_pred_2, y_true_2)

        cm = confusion_matrix(y_pred_match_1, y_pred_match_2)
        a,b,c,d = cm.ravel()
        if b+c == 0:
            return np.nan
        chi_stat = (abs(b-c)-1)**2/(b+c)
        p_value = 1 - chi2.cdf(chi_stat,1)
        self.chi2_stats.append(chi_stat)
        self.p_values.append(p_value)
        return chi_stat, p_value, (a,b,c,d)

    def get_results(self):
        return self.chi2_stats, self.p_values

class MultipleMcNemarTest(StatTest):
    def __init__(self):
        self.chi2_stats = []
        self.p_values = []

    def eval(self, y_pred_prob_1, y_pred_prob_2)-> np.array:
        y_true_1 = y_pred_prob_1[:,0]
        y_true_2 = y_pred_prob_2[:,0]
        y_pred_1 = from_pred_prob_to_pred_class(y_pred_prob_1[:,1:])
        y_pred_2 = from_pred_prob_to_pred_class(y_pred_prob_2[:,1:])

        if not  np.array_equal(y_true_1,y_true_2):
            raise ValueError("True labels are not the same")
        
        y_pred_match_1 = np.equal(y_pred_1, y_true_1)
        y_pred_match_2 = np.equal(y_pred_2, y_true_2)

        cm = confusion_matrix(y_pred_match_1, y_pred_match_2)
        a,b,c,d = cm.ravel()
        if b+c == 0:
            return np.nan
        chi_stat = (abs(b-c)-1)**2/(b+c)
        p_value = 1 - chi2.cdf(chi_stat,1)
        self.chi2_stats.append(chi_stat)
        self.p_values.append(p_value)
        return chi_stat, p_value, (a,b,c,d)

    def get_results(self):
        return self.chi2_stats, self.p_values


class WilcoxonTest(StatTest):
    pass

if __name__ == "__main__":
    n_data = 100
    n_classes = 3
    y_true = np.random.randint(0,1,size=n_data)
    tester = McNemarTest()
    y_pred_prob1= np.random.rand(n_data, n_classes)
    y_pred_prob2= np.random.rand(n_data, n_classes)
    y_pred_prob1 = np.concatenate([y_true[:,None], y_pred_prob1], axis=1)
    y_pred_prob2 = np.concatenate([y_true[:,None], y_pred_prob2], axis=1)
    tester.eval(y_pred_prob1, y_pred_prob2)
    print(tester.get_results())