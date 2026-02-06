import sys

import pandas as pd
import math as m
import nltk

class TextClassifier:

    def __init__(self, csv_file_name):
        self.model_params = pd.read_csv(csv_file_name, index_col=0)

    def compute_probability(self, text_string):
        # Tokenize the text
        tokens = nltk.word_tokenize(text_string)
        # Get all labels
        labels = self.get_all_possible_labels()
        # Get all features
        features = self.get_all_possible_features()

        # Scoring the labels
        labels_score = {}
        for label in labels:
            labels_score[label] = 0
            for token in tokens:
                if token in features:
                    weight = self.model_params.loc[token][label]
                    labels_score[label] += weight

        # Calculating sum of exp(score) of all labels
        sum_exp = 0
        for label in labels:
            sum_exp += m.exp(labels_score[label])

        # Calcuting probability
        prob_label = {}
        for label in labels:
            prob_label[label] = round(m.exp(labels_score[label])/sum_exp, 2)

        return prob_label


    def get_all_possible_features(self):
        return self.model_params.index.to_list()

    def get_all_possible_labels(self):
        return self.model_params.columns.to_list()

    def classify(self, text_string):
        # Get probability
        probs = self.compute_probability(text_string)

        # Get key with the max value
        max_key = max(probs, key=probs.get)
        return max_key



if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print('usage:\tpython logistic_regression.py <model_file>')
        sys.exit(0)
    model_file_name = sys.argv[1]
    model = TextClassifier(model_file_name)
    print(model.get_all_possible_labels())
    print(model.get_all_possible_features())
    print(model.compute_probability('I hate dust'))
    print(model.classify('I hate dust'))
