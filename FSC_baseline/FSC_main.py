import os
import json
import pandas as pd
import random
import copy
import numpy as np
from semspaces.space import SemanticSpace
from datetime import datetime
from neighbors import get_levenshtein_neighbours
from fsc_ld_linux import levenshtein_fsc

random_baseline = True

if __name__ == '__main__':

    #ask for the path to the map in which all file will be stored
    fsc_dir_rnd = input("Please provide a path to a map where all output will be stored\n")
    if not os.path.exists(fsc_dir_rnd):
        raise ValueError("This directory does not exist, please provide a valid path!")

    #ask for all files needed for the computations
    reference_w2v_space_filebase = input("Please provide the path to the directory where all reference w2v spaces are stored\n") 
    if not os.path.exists(reference_w2v_space_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    target_w2v_space_filebase = input("Please provide the path to the directory where all target w2v spaces are stored\n") 
    if not os.path.exists(target_w2v_space_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    produced_reference_w2v_filebase = input("Please provide the path to the directory where all produced reference w2v spaces are stored\n") 
    if not os.path.exists(produced_reference_w2v_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    reference_wordcounts_filebase = input("Please provide the path to the directory where all reference wordcounts are stored\n") 
    if not os.path.exists(reference_wordcounts_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    target_wordcounts_filebase = input("Please provide the path to the directory where all target wordcounts are stored\n") 
    if not os.path.exists(target_wordcounts_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")
    

    reference_w2v_filelist = os.listdir(reference_w2v_space_filebase)
    target_w2v_filelist = os.listdir(target_w2v_space_filebase)
    produced_reference_w2v_filelist = os.listdir(produced_reference_w2v_filebase)
    reference_wordcounts_filelist = os.listdir(reference_wordcounts_filebase)
    target_wordcounts_filelist = os.listdir(target_wordcounts_filebase)

    for i in range(len(reference_w2v_filelist)):

        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computation for batch {} out of {}. \n".format(i+1, len(reference_w2v_filelist)))
        )

        embedding_space = SemanticSpace.from_csv(os.path.join(target_w2v_space_filebase, target_w2v_filelist[i]), prenorm = True)
        w2v_words = embedding_space.included_words()

        reference_space = SemanticSpace.from_csv(os.path.join(reference_w2v_space_filebase, reference_w2v_filelist[i]), prenorm = True)
        w2v_reference = reference_space.included_words()

        child_produced_space = SemanticSpace.from_csv(os.path.join(produced_reference_w2v_filebase, produced_reference_w2v_filelist[i]), prenorm = True)
        w2v_child_produced = child_produced_space.included_words()

        wordcount = json.load(open(os.path.join(target_wordcounts_filebase, target_wordcounts_filelist[i])))
        reference_wordcount = json.load(open(os.path.join(reference_wordcounts_filebase, reference_wordcounts_filelist[i])))
        age_bin = wordcount["Age_in_Months"]


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

        print('The target vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(target_vocab)))
        print('The full reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(full_reference_vocab)))
        print('The child-produced reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(produced_reference_vocab)))
        print('The most used reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(most_used_reference_vocab)))
        
        #find neighbors (levenshstein) for orthographic froms
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started retrieving Levenshtein distance neighbors for orthographic forms.")
        )
        #full_ortho2neighbors_ld = get_levenshtein_neighbours(target_vocab, full_reference_vocab)
        produced_ortho2neighbors_ld = get_levenshtein_neighbours(target_vocab, produced_reference_vocab)
        #most_ortho2neighbors_ld = get_levenshtein_neighbours(target_vocab, most_used_reference_vocab)
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done retrieving Levenshtein distance neighbors for orthographic forms.")
        )

        if random_baseline:

            n_subsamples = 100
            seeds = random.sample(range(0, 100000000), n_subsamples)
            print(datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S: Started computing FSC from {} random permutations of the embeddings...".format(n_subsamples)
            ))

            # COMPUTE FSC MEASURES FROM RANDOM PERMUTATIONS OF THE WORD EMBEDDINGS, REPEAT 1000 TIMES AND SAVE MEASURES TO FILE
            random_reference = copy.deepcopy(reference_space)
            if not os.path.exists(fsc_dir_rnd):
                os.makedirs(fsc_dir_rnd)

            filename = "fsc_random_baseline" + str(age_bin) + ".csv"

            targets = list(target_vocab)
            final_df = pd.DataFrame(
                data = targets,
                columns = ["word"]
            )

            for i, seed in enumerate(seeds):
                np.random.seed(seed)
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started permutation {} of {}...".format(i + 1, n_subsamples)))
                random_reference.vectors = np.random.permutation(random_reference.vectors)

                full_t2OSC_ld_rnd = levenshtein_fsc(full_ortho2neighbors_ld, random_embeddings, reference_space)
                produced_t2OSC_ld_rnd = levenshtein_fsc(produced_ortho2neighbors_ld, embedding_space, random_reference)
                most_t2OSC_ld_rnd = levenshtein_fsc(most_ortho2neighbors_ld, random_embeddings, reference_space)

                values = []

                for word in targets:
                    values.append(
                        [word, full_t2OSC_ld_rnd[word], produced_t2OSC_ld_rnd[word], most_t2OSC_ld_rnd[word]] 
                    )

                full = "full_OSC_rnd" + str(i+1)
                produced = "produced_OSC_rnd" + str(i+1)
                most = "most_OSC_rnd" + str(i+1)
                
                df = pd.DataFrame(
                    data = values,
                    columns = ["word", full, produced, most]
                )

                final_df = pd.merge(final_df, df, on = "word")

            final_df.to_csv(os.path.join(fsc_dir_rnd, filename), index = False, sep = ';')

    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done with computation for batch {} out of {}. \n".format(i+1, len(reference_w2v_filelist)))
    )
            
            



    
