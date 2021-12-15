__author__ = 'Gcassani'

"""This module creates a dictionary containing morphological and phonological information from the CELEX database,
   at both the token and type level, to be used to recode transcribed speech into phonologically and morphologically 
   richer representations (can be called from command line)"""

import os
import argparse
from celex.get import get_celex_dictionary


def main():

    celex_dir = input("Please prove the path to the Celex directory (file containing eml.cd, emw.cd, elp.cd and epw.cd files) \n")
    get_celex_dictionary(celex_dir, reduced = False)


########################################################################################################################


if __name__ == '__main__':

    main()
