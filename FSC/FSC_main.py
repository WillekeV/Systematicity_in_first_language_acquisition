import itertools
import os
import json
import pickle
import pandas as pd
import random
import copy
import numpy as np
from semspaces.space import SemanticSpace
from datetime import datetime
from celex import get_celex_coverage
from neighbors import get_levenshtein_neighbours
from fsc_ld import levenshtein_fsc #use fsc_ld_linux if not working on windows
from semantic import neighborhood_density
from resources import aoa, concreteness, valence, morpholex
from compute_old20 import calculate_old20

force_recomputation = False
random_baseline = True

def read(path):

    """
    :param path:            the path to the csv file containing the SUBTLEX-US frequency values
    :return:                a dict measures words to frequency values
    """

    df = pd.read_csv(path, header=0)
    df["word"] = df["Word"].str.lower()
    word2freq = pd.Series(df['SUBTLWF'].values, index=df['word']).to_dict()

    return word2freq

def write_df(targets, out_path, full_d_OSC_ld, produced_d_OSC_ld, most_d_OSC_ld, full_d_snd, produced_d_snd, most_d_snd, wordcount, measure, 
             reference_size_full, reference_size_produced, reference_size_most, embedding_type, w2concr, w2val, w2morph, w2aoa, aoa_produced,
             full_old20, produced_old20, most_old20): 

    """
    :param targets:             iterable, containing target words
    :param out_path:            str, indicating the path where to save the combined df
    :param full_d_OSC_ld:       dict, maps target words to OSC values computed using levenshtein distance neighbors on the whole reference vocab (systematicity measure)
    :param produced_d_OSC_ld:   dict, maps target words to OSC values computed using levenshtein distance neighbors on the part of the reference vocab which was also produced by the child (systematicity measure)
    :param most_d_OSC_ld:       dict, maps target words to OSC values computed using levenshtein distance neighbors on the 20% most used words in the reference vocab (systematicity measure)
    :param d_target_phon:       dict, maps target words to corresponding phonological forms
    :param d_morph:             dict, maps target words to morphological status
    :param full_d_snd:          dict, maps target words to semantic neighborhood density values computed on the whole reference vocab
    :param produced_d_snd:      dict, maps target words to semantic neighborhood density values computed on the part of the reference vocab which was also produced by the child
    :param most_d_snd:          dict, maps target words to semantic neighborhood density values computed on the 20% most used words in the reference vocab 
    :param wordcount            dict, maps target words to their word count and also contains the age bin and if the speech is directed or produced
    :param measure              specifies which method was used to get the systematicity measure
    :param reference_size       specifies how many different words are present in the reference vocab
    :param embedding_type       specifies by which method the embedding space was created
    :param w2concr              concreteness values of words (if available)
    :param w2val                valance values of words (if available)
    :param w2old                old values of words (if available)
    :param w2morph              morpholexic complexity values of words (if available)
    :param w2aoa                AoA values for words (if available)
    :param aoa_produced         AoA values for words based on when the child uttered the word at least 2 times
    :param aoa_directed         AoA values for words based on when the child heard the word at least 2 times
    """

    age_mapping_dict = {"24":0, "30":1, "36":2, "42":3, "48":4, "54":5, "60": 6, "72":7, "72+":8, "later": 9}

    values = []
    for word in targets:
        if word in wordcount:
            wordcount_word = wordcount[word]
        else:
            wordcount_word = 1

        if word in w2concr:
            w2concr_word = w2concr[word]
        else:
            w2concr_word = '-'

        if word in w2val:
            w2val_word = w2val[word]
        else:
            w2val_word = '-'

        if word in w2aoa:
            w2aoa_word = w2aoa[word]
            try:
                w2aoa_word_months = int(w2aoa_word) * 12
                if w2aoa_word_months <= 24:
                    w2aoa_word_months = 24
                elif w2aoa_word_months > 24 and w2aoa_word_months <= 30:
                    w2aoa_word_months = 30
                elif w2aoa_word_months > 30 and w2aoa_word_months <= 36:
                    w2aoa_word_months = 36
                elif w2aoa_word_months > 36 and w2aoa_word_months <= 42:
                    w2aoa_word_months = 42
                elif w2aoa_word_months > 42 and w2aoa_word_months <= 48:
                    w2aoa_word_months = 48
                elif w2aoa_word_months > 54 and w2aoa_word_months <= 60:
                    w2aoa_word_months = 60
                elif w2aoa_word_months > 60 and w2aoa_word_months <= 72:
                    w2aoa_word_months = 72
                elif w2aoa_word_months > 72 and w2aoa_word_months <= 120:
                    w2aoa_word_months = "72+"
                elif w2aoa_word_months > 120:
                    w2aoa_word_months = "later"
            except ValueError:
                w2aoa_word_months = '-'
        else:
            w2aoa_word = '-'
            w2aoa_word_months = "-"

        if word in aoa_produced:
            aoa_produced_word = aoa_produced[word]
        else:
            aoa_produced_word = "later"

        if w2aoa_word_months != '-':
            if w2aoa_word_months == 'later':
                age_bin_difference_aoa = 8
            else:
                age_bin_difference_aoa = age_mapping_dict[str(w2aoa_word_months)] - age_mapping_dict[str(age_bin)]
        else:
            age_bin_difference_aoa = '-'

        if aoa_produced_word == "later":
            age_bin_difference_produced = 8
        else: 
            age_bin_difference_produced = age_mapping_dict[str(aoa_produced_word)] - age_mapping_dict[str(age_bin)]
        

        values.append(
            [measure, word, len(word), wordcount_word, full_d_OSC_ld[word], produced_d_OSC_ld[word], most_d_OSC_ld[word], full_d_snd[word], 
             produced_d_snd[word], most_d_snd[word], reference_size_full, reference_size_produced, reference_size_most, wordcount["Age_in_Months"], embedding_type,
             w2concr_word, w2val_word, w2morph[word], w2aoa_word, w2aoa_word_months, age_bin_difference_aoa, 
             aoa_produced_word, age_bin_difference_produced, full_old20[word], produced_old20[word], most_old20[word]]
        )


    final_df = pd.DataFrame(
        data=values,
        columns=["measure", "word", "length", "wordcount", "full_OSC_ld", "produced_OSC_ld", "most_OSC_ld", "full_snd",
                 "produced_snd", "most_used_snd", "reference_size_full", "reference_size_produced", "reference_size_most", "age_bin", "embedding_type",
                 "concreteness", "valence", "morph", "AoA", "AoA_in_months", "AoA_age_bin_difference",
                "AoA_child_produced", "age_bin_difference_produced", "full_old20", "produced_old20", "most_old20"]
    )

    final_df.to_csv(out_path, index=False, sep=';')

if __name__ == '__main__':

    #ask for the target embedding space file. First line of the file should be the size (for example 10000 300).
    embedding_space_file = input("Please provide the path file in which the target embedding space is stored.\n") 
    if not os.path.exists(embedding_space_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the reference embedding space file. First line of the file should be the size (for example 10000 300).
    reference_space_file = input("Please provide the path file in which the reference embedding space is stored.\n") 
    if not os.path.exists(reference_space_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the child-produced reference embedding space file. First line of the file should be the size (for example 10000 300).
    child_produced_space_file = input("Please provide the path file in which the child-produced-reference embedding space is stored.\n") 
    if not os.path.exists(child_produced_space_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the file containing a dictionary of wordcounts for all words in the target vocab. It should also include the age of the child and if the speech is directed or produced.
    wordcount_file = input("Please provide the path to the file containing wordcounts for all words in the target embedding space. \n")
    if not os.path.exists(wordcount_file):
        raise ValueError("This directory does not exist, please provide a valid path!") 

    #ask for the file containing a dictionary of wordcounts for all the words in the reference vocab.
    wordcount_file_reference = input("Please provide the path to the file containing wordcounts for all words in the reference embedding space. \n")
    if not os.path.exists(wordcount_file_reference):
        raise ValueError("This directory does not exist, please provide a valid path!")

    #ask for the morpholex file
    morpholex_file = C:/Users/wgvan/Documents/Universiteit_Nijmegen/jaar2/thesis/Project_Code/FSC/MorphoLEX_en.xlsx
    if not os.path.exists(morpholex_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the Celex dictionary file
    celex_file = input("Please provide the path to the file containing the celex dictionary.\n") 
    if not os.path.exists(celex_file):
        raise ValueError("The file does not exist, please provide a valid file!")
        
    aoa_produced_file = input("Please provide the path to the file containing the mapping to when the child first produced the word.\n") 
    if not os.path.exists(aoa_produced_file):
        raise ValueError("The file does not exist, please provide a valid file!")
        
    aoa_file = input("Please provide the path to the file containing the mapping to AoA ratings.\n") 
    if not os.path.exists(aoa_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    concreteness_file = input("Please provide the path to the file containing the mapping to concreteness ratings.\n") 
    if not os.path.exists(concreteness_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    valence_file = input("Please provide the path to the file containing the mapping to valence ratings.\n") 
    if not os.path.exists(valence_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the path to the map in which all file will be stored
    fsc_dir = input("Please provide a path to a map where all output will be stored") 
    if not os.path.exists(fsc_dir):
        raise ValueError("This directory does not exist, please provide a valid path!")

    #Loading the data
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started loading the data... \n"))

    embedding_space = SemanticSpace.from_csv(embedding_space_file, prenorm=True)
    w2v_words = embedding_space.included_words()

    reference_space = SemanticSpace.from_csv(reference_space_file, prenorm = True)
    w2v_reference = reference_space.included_words()

    child_produced_space = SemanticSpace.from_csv(child_produced_space_file, prenorm = True)
    w2v_child_produced = child_produced_space.included_words()

    celex = json.load(open(celex_file))

    wordcount = json.load(open(wordcount_file))
    reference_wordcount = json.load(open(wordcount_file_reference))
    age_bin = wordcount["Age_in_Months"]

    aoa_produced = json.load(open(aoa_produced_file))

    aoa_words, aoa_norms = aoa.read(aoa_file)
    w2aoa = pd.Series(aoa_norms["Rating.Mean"].values, index=aoa_norms["Word"]).to_dict()

    w2concr = concreteness.read(concreteness_file)
    w2val = valence.read(valence_file)

    mono = list(morpholex.read_mono(morpholex_file))
    poly = list(morpholex.read_poly(morpholex_file))
    mono_inflected = list(morpholex.read_mono_inflected(morpholex_file))

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done loading data. \n"))

    #obtain target vocabulary: words which are in the current vocab, but not in the reference one (and occur at least twice)
    precursor_target_vocab = set(w2v_reference.symmetric_difference(w2v_words))
    target_vocab_list = []
    for word in precursor_target_vocab:
        try:
            if wordcount[word] >= 2:
                target_vocab_list.append(word)
        except KeyError:
            continue
    target_vocab = set(target_vocab_list)
    #target_vocab = set(itertools.islice(target_vocab_list, 0, 16))

    #obtain the 20% most frequent words in the reference vocab
    reference_wordcount_list = list(reference_wordcount.values())
    reference_wordcount_list.pop()
    reference_wordcount_list.pop()
    reference_wordcount_list.sort(reverse=True)
    most_used = round(len(reference_wordcount_list)/5)
    most_used_threshold = reference_wordcount_list[most_used]
    most_used_reference_vocab = []
    for word in w2v_reference:
        try: 
            if reference_wordcount[word] >= most_used_threshold:
                most_used_reference_vocab.append(word)
        except KeyError:
            continue
    
    #obtain all words from the reference vocab, which have been uttered by the child before
    produced_reference_vocab = list(w2v_reference.intersection(w2v_child_produced))

    #obtain the full reference vocab
    full_reference_vocab = list(w2v_reference)

    #Assign boolean value to each word for which all variables are available depending on their morphological complexity
    w2morph = morpholex.compute_morph_complexity(list(target_vocab), mono, mono_inflected, poly)

    print('The target vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(target_vocab)))
    print('The full reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(full_reference_vocab)))
    print('The child-produced reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(produced_reference_vocab)))
    print('The most used reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(most_used_reference_vocab)))

    #Compute semantic neighborhood density
    fullsndfile = "fulltarget2snd" + str(age_bin) + ".json"
    snd_path = os.path.join(fsc_dir, fullsndfile)
    try:
        if force_recomputation:
            raise FileNotFoundError()
        full_t2snd = json.load(open(snd_path, "rb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} found and loaded.".format(snd_path)))
    except FileNotFoundError:
        full_t2snd = neighborhood_density(embedding_space, reference_space, target_vocab, full_reference_vocab)
        json.dump(full_t2snd, open(snd_path, 'w'))

    producedsndfile = "childproducedtarget2snd" + str(age_bin) + ".json"
    produced_snd_path = os.path.join(fsc_dir, producedsndfile)
    try:
        if force_recomputation:
            raise FileNotFoundError()
        produced_t2snd = json.load(open(produced_snd_path, "rb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} found and loaded.".format(produced_snd_path)))
    except FileNotFoundError:
        produced_t2snd = neighborhood_density(embedding_space, reference_space, target_vocab, produced_reference_vocab)
        json.dump(produced_t2snd, open(produced_snd_path, 'w'))

    mostsndfile = "mostusedtarget2snd" + str(age_bin) + ".json"
    most_snd_path = os.path.join(fsc_dir, mostsndfile)
    try:
        if force_recomputation:
            raise FileNotFoundError()
        most_t2snd = json.load(open(most_snd_path, "rb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} found and loaded.".format(most_snd_path)))
    except FileNotFoundError:
        most_t2snd = neighborhood_density(embedding_space, reference_space, target_vocab, most_used_reference_vocab)
        json.dump(most_t2snd, open(most_snd_path, 'w'))

    #compute old20 values for each word
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing old20 values for all words.")
    )
    full_old20 = calculate_old20(target_vocab, full_reference_vocab)
    produced_old20 = calculate_old20(target_vocab, produced_reference_vocab)
    most_old20 = calculate_old20(target_vocab, most_used_reference_vocab)
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing old20 values for all words.")
    )

    
    #find neighbors (levenshstein) for orthographic froms
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started retrieving Levenshtein distance neighbors for orthographic forms.")
    )
    full_ortho2neighbors_ld = get_levenshtein_neighbours(target_vocab, full_reference_vocab)
    produced_ortho2neighbors_ld = get_levenshtein_neighbours(target_vocab, produced_reference_vocab)
    most_ortho2neighbors_ld = get_levenshtein_neighbours(target_vocab, most_used_reference_vocab)
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done retrieving Levenshtein distance neighbors for orthographic forms.")
    )
    

    #Try to fetch FSC values (ortho, computed using Levenshtein distance, neighbors) from file. If the file does not exist, compute values
    full_ldfile = "full_OSC_ld" + str(age_bin) + ".pkl"
    osc_ld_path = os.path.join(fsc_dir, full_ldfile)
    try:
        if force_recomputation:
            raise FileNotFoundError()
        full_t2OSC_ld = pickle.load(open(osc_ld_path, "rb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File found and loaded."))
    except FileNotFoundError:
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing Levenshtein OSC..."))
        full_t2OSC_ld = levenshtein_fsc(full_ortho2neighbors_ld, embedding_space, reference_space)
        pickle.dump(full_t2OSC_ld, open(osc_ld_path, "wb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done."))
        print()

    produced_ldfile = "produced_OSC_ld" + str(age_bin) + ".pkl"
    produced_osc_ld_path = os.path.join(fsc_dir, produced_ldfile)
    try:
        if force_recomputation:
            raise FileNotFoundError()
        produced_t2OSC_ld = pickle.load(open(produced_osc_ld_path, "rb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File found and loaded."))
    except FileNotFoundError:
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing Levenshtein OSC..."))
        produced_t2OSC_ld = levenshtein_fsc(produced_ortho2neighbors_ld, embedding_space, reference_space)
        pickle.dump(produced_t2OSC_ld, open(produced_osc_ld_path, "wb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done."))
        print()

    most_ldfile = "mostused_OSC_ld" + str(age_bin) + ".pkl"
    most_osc_ld_path = os.path.join(fsc_dir, most_ldfile)
    try:
        if force_recomputation:
            raise FileNotFoundError()
        most_t2OSC_ld = pickle.load(open(most_osc_ld_path, "rb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File found and loaded."))
    except FileNotFoundError:
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing Levenshtein OSC..."))
        most_t2OSC_ld = levenshtein_fsc(most_ortho2neighbors_ld, embedding_space, reference_space)
        pickle.dump(most_t2OSC_ld, open(most_osc_ld_path, "wb"))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done."))
        print()

    #write measures to file for subsequent analysis
    measure = "FSC"
    size_of_reference_full = len(full_reference_vocab)
    size_of_reference_produced = len(produced_reference_vocab)
    size_of_reference_most = len(most_used_reference_vocab)
    embedding_type = "w2v"
    filename = "fsc_measures" + str(age_bin) + ".csv"
    write_df(target_vocab, os.path.join(fsc_dir, filename), full_t2OSC_ld, produced_t2OSC_ld, most_t2OSC_ld, full_t2snd, produced_t2snd, most_t2snd,
             wordcount, measure, size_of_reference_full, size_of_reference_produced, size_of_reference_most, embedding_type, w2concr, w2val, w2morph, w2aoa, aoa_produced,
             full_old20, produced_old20, most_old20) 
