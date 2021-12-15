__author__ = 'GCassani'

"""Function to encode the corpus into phonetic cues and lexical outcomes (can be called from command line)"""

import os
from celex.utilities.dictionaries import tokens2ids
from corpus.encoder import corpus_encoder

#celex_dir = "" # path to Celex directory
#pos_mapping = "" # File containing the PoS mapping from the CHILDES tags to the CELEX tags (one line should look like: "Adj A")
#in_file = "" # File containing the corpus to be used as input (as .json), with one list of lists containing sentences in token form and the other a list of list containing sentences in lemma + PoS tag
#separator = "~" # the separator between the lemma and the PoS tag in the in_file
#outcomes = 'tokens' # specify the lexical outcome of the outfile, either 'tokens' or 'lemmas'
#uniphones = True # specify if uniphones need to be encoded
#diphones = False # specify if diphones need to be encoded
#triphones = False # specify if triphones need to be encoded
#syllables = False # specify if syllables need to be encoded
#stress_marker = False # specify if stressmaker needs to be encoded
#boundaries = True # specify whether word boundaries are to be considered when training on utterances
#reduced = True # specify if reduced vowels are to be considered when extracting CELEX phonetic forms

def check_arguments(uniphones, diphones, triphones, syllables, celex_dir, pos_mapping, in_file):

    if not (uniphones or diphones or triphones or syllables):
        raise ValueError('No specified phonetic encoding! Provide at least one of the following options: -u, -d, -t, -s')

    if not os.path.exists(celex_dir):
        raise ValueError("The Celex directory you provided doesn't exist. Provide a valid path.")

    if not os.path.exists(pos_mapping):
        raise ValueError("The file containing the mapping from CHILDES to Celex PoS tags isn't valid."
                         "Please provide a valid path.")

    if not os.path.exists(in_file) or not in_file.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")


########################################################################################################################


def main():
    uniphones = False # specify if uniphones need to be encoded
    diphones = False # specify if diphones need to be encoded
    triphones = False # specify if triphones need to be encoded
    syllables = False # specify if syllables need to be encoded
    stress_marker = False # specify if stressmaker needs to be encoded
    boundaries = True # specify whether word boundaries are to be considered when training on utterances
    reduced = True # specify if reduced vowels are to be considered when extracting CELEX phonetic forms

    celex_dir = input("Please provide the path to the Celex directory (file containing eml.cd, emw.cd, elp.cd and epw.cd files) \n")
    pos_mapping = input("Please provide the path to file containing the PoS mapping from the CHILDES tags to the CELEX tags (one line should look like: 'Adj A') \n") 
    in_file = input("Please provide the path to the corpus to be used as input (as .json) with one list of lists containing sentences in token form and the other a list of list containing sentences in lemma + PoS tag \n") 
    separator = "~" # the separator between the lemma and the PoS tag in the in_file
    outcomes = input("Please specify if the outcomes should be either 'tokens' or lemmas' (type either tokens or lemmas) \n")

    phones = input("specify if either uniphones, diphones, triphones or syllables need to be encoded (type uni, di , tri or syl) \n")
    if phones == "uni":
        uniphones = True
        diphones = False
        triphones = False
        syllables = False
    elif phones == "di":
        diphones = True
        uniphones = False
        triphones = False
        syllables = False
    elif phones == "tri":
        triphones == True
        diphones = False
        uniphones = False
        syllables = False
    elif phones == "syl":
        syllables = True
        diphones = False
        triphones = False
        uniphones = False
    else:
        raise ValueError('No specified phonetic encoding! Provide at least one of the following options: uni, di, tri or syl')
    
    stress = input("specify if a stressmarker needs to be encoded, type either yes or no \n")
    if stress == "yes":
        stress_marker = True
    elif stress == "no":
        stress_marker = False
    else:
        raise ValueError("not specified if stressmarker should be encoded or not! Provide one of the following options: yes or no")

    boundary = input("specify if a word-boundary needs to be encoded, type either yes or no \n")
    if boundary == "yes":
        boundaries = True
    elif boundary == "no":
        boundaries = False
    else:
        raise ValueError("not specified if word-boundaries should be encoded or not! Provide one of the following options: yes or no")

    reduce = input("specify if reduced word-forms should be used when available, type either yes or no \n")
    if reduce == "yes":
        reduced = True
    elif reduce == "no":
        reduced = False
    else:
        raise ValueError("not specified if reduced word forms shpuld be used or not! Provide one of the following options: yes or no")
    

    check_arguments(uniphones, diphones, triphones, syllables, celex_dir, pos_mapping, in_file)

    corpus_encoder(in_file, celex_dir, pos_mapping, separator, reduced, outcomes,
                   uniphones, diphones, triphones, syllables,
                   stress_marker,
                   boundaries)


########################################################################################################################


if __name__ == '__main__':

    main()
