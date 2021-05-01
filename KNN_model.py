import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd
from collections import Counter

def get_class(arr):
    #arr = np.sort(arr)
    result = Counter(arr).most_common(1)
    return result

class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        """
        if X.ndim == 1: X = np.reshape(X, (1,X.shape[0]))
        self.X_train = X
        self.Y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.
        
        """

        # reshape dim to 2 if np shape is 1
        if X.ndim == 1: X = np.reshape(X, (1,X.shape[0]))
        
        #Minkowski distance calculation - temp[i,j] is the distance between i and j
        temp = np.power(np.sum(np.power(np.abs(X[:,np.newaxis]-self.X_train), self.p), axis = 2), 1/self.p)
        
        # create new array containing Minkowski distance as the real part and the label as the imaginary - used in argsort
        temp = temp + 1j*self.Y_train
        
        # index of the k closest neighbors, sorted by distance(real part - makes the closest neighbor first) and then
        # sorted by label (imaginary  part - if there is a distance tie sorts by lexicographic ordering)
        # np.argsort - on complex numbers sorts by real numbers and then by imaginary numbers.
        idx = temp.argsort()[:,:self.k]
        
        # in Counter.most_common "Elements with equal counts are ordered in the order first encountered"
        # sort made the most_common function encounter neighbors based on the tie breaking rules
        prediction = np.apply_along_axis(get_class, 1, self.Y_train[idx[:,:self.k]])
        return prediction[:,0,0]
    

def main():

    print("*" * 20)
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
