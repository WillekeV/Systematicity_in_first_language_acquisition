__author__ = 'GCassani'

"""Function to estimate cue-outcome connections given an input corpus encoded as lists of cues matched to sets of
   outcomes using naïve discriminative learning (can be called from command line)"""

import os
import glob
from rescorla_wagner.ndl import ndl


def main():

    input_corpus = input("Specify the corpus to be used as input, consisting of lists of phonetic cues or lexical inputs"
                             "paired to sets of lexical outcomes (the file needs to be encoded as .json). \n")

    if not os.path.exists(input_corpus):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    longitudinal = False #input("Specify whether to work in a longitudinal design or not (default: False).")

    alpha = 0.01 #input("Specify the value of the alpha parameter (default: 0.01).")

    beta = 0.01 # input("Specify the value of the beta parameter (default: 0.01).")

    lam = 1.0 #input("Specify the value of the lambda parameter (default: 1.0).")

  

    ndl(input_corpus, longitudinal, alpha, beta, lam)

########################################################################################################################


if __name__ == '__main__':

    main()
