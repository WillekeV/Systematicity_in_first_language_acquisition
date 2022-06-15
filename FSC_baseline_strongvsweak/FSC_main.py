import os
import json
import pandas as pd
import random
import copy
import numpy as np
import pickle
import string
from semspaces.space import SemanticSpace
from datetime import datetime
from neighbors import get_levenshtein_neighbours
from fsc_ld_linux import levenshtein_fsc
from fsc_ld_full_linux import levenshtein_fsc_full

if __name__ == '__main__':
       
    #ask for the path to the map in which all file will be stored
    fsc_dir_strongvsweak = input("Please provide a path to a map where all output will be stored\n")
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

    reference_w2v_filelist = sorted(os.listdir(reference_w2v_space_filebase))
    target_w2v_filelist = sorted(os.listdir(target_w2v_space_filebase))
    produced_reference_w2v_filelist = sorted(os.listdir(produced_reference_w2v_filebase))
    reference_wordcounts_filelist = sorted(os.listdir(reference_wordcounts_filebase))
    target_wordcounts_filelist = sorted(os.listdir(target_wordcounts_filebase))

    def write_df(targets, out_path, OSC_ld):

        values = []
        for word in targets:
            values.append(
                [word, OSC_ld[word]]
            )

        final_df = pd.DataFrame(
            data = values,
            columns = ["word", "OSC_ld"]
        )

        final_df.to_csv(out_path, index = False, sep = ';')

    for i in range(len(reference_w2v_filelist)):

        print(
            datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S: Started computation for batch {} out of {}. \n".format(
                    i+1, len(reference_w2v_filelist
                             )
                )
            )
        )

        reference_space_file = os.path.join(reference_w2v_space_filebase, reference_w2v_filelist[i])
        reference_space = SemanticSpace.from_csv(
            reference_space_file, prenorm = True
        )
        w2v_reference = reference_space.included_words()

        reference_wordcount = json.load(
            open(os.path.join(reference_wordcounts_filebase, reference_wordcounts_filelist[i]))
        )
        age_bin_reference = reference_wordcount["Age_in_Months"]

        vocab = set(w2v_reference)

        print('The vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(vocab)))

        #check if the file containing all fsc values already exists and open it, if not, compute fsc values and save this to file
        filename_full = "full_fsc_measures_strongvsweak_" + str(age_bin_reference) + ".csv"
        out_path = os.path.join(fsc_dir_strongvsweak, filename_full)
        try:
            systematicity_file = pd.read_csv(out_path, sep = ";")
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} found and loaded.".format(out_path)))
        except FileNotFoundError:   
            #find neighbors (levenshstein) for orthographic froms
            print(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started retrieving Levenshtein distance neighbors for orthographic forms for full reference.")
            )
            ortho2neighbors_ld = get_levenshtein_neighbours(vocab, vocab)
            print(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done retrieving Levenshtein distance neighbors for orthographic forms for full reference.")
            )

            #find levenshtein distances for all words in the reference vocab, so that they all have a measure of systematicity
            full_ldfile = "full_OSC_ld" + str(age_bin_reference) + ".pkl"
            osc_ld_path = os.path.join(fsc_dir_strongvsweak, full_ldfile)
            try:
                t2OSC_ld = pickle.load(open(osc_ld_path, "rb"))
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} found and loaded.".format(osc_ld_path)))
            except FileNotFoundError:
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing Levenshtein OSC..."))
                t2OSC_ld = levenshtein_fsc_full(ortho2neighbors_ld, reference_space)
                pickle.dump(t2OSC_ld, open(osc_ld_path, "wb"))
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done."))
                print()

            write_df(vocab, out_path, t2OSC_ld)
            systematicity_file = pd.read_csv(out_path, sep = ";")
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} created.".format(out_path)))

        embedding_space = SemanticSpace.from_csv(
            os.path.join(target_w2v_space_filebase, target_w2v_filelist[i]), prenorm = True
        )
        w2v_words = embedding_space.included_words()

        wordcount = json.load(open(os.path.join(target_wordcounts_filebase, target_wordcounts_filelist[i])))
        age_bin = wordcount["Age_in_Months"]

        child_produced_space = SemanticSpace.from_csv(
            os.path.join(produced_reference_w2v_filebase, produced_reference_w2v_filelist[i]), prenorm = True
            )
        w2v_child_produced = child_produced_space.included_words()

        systematicity_scores = systematicity_file.sort_values(by=['OSC_ld'], ascending = False)

        percentage_list = [0.1, 0.3, 0.5, 0.7, 0.9]

        #obtain target vocabulary: words which are in the current vocab, but not in the reference one
        # (and occur at least twice)
        precursor_target_vocab = set(w2v_reference.symmetric_difference(w2v_words))
        target_vocab_list = []
        for word in precursor_target_vocab:
            try:
                if wordcount[word] >= 2:
                    target_vocab_list.append(word)
            except KeyError:
                continue
        target_vocab = set(target_vocab_list)

        #obtain all words from the reference vocab, which have been uttered by the child before
        produced_reference_vocab = list(w2v_reference.intersection(w2v_child_produced))

        print('The target vocabulary for retrieving form-based neighbors consists of {} words.'.format(
            len(target_vocab))
        )

        print('The child-produced reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(
            len(produced_reference_vocab))
        )

        #find neighbors (levenshstein) for orthographic froms
        print(
            datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S: Started retrieving Levenshtein distance neighbors for orthographic forms."
            )
        )
        produced_ortho2neighbors_ld = get_levenshtein_neighbours(target_vocab, produced_reference_vocab)
        print(
            datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S: Done retrieving Levenshtein distance neighbors for orthographic forms."
            )
        )

        for n in range(len(percentage_list)):
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing for percentage batch {} of {}...".format(n + 1, len(percentage_list))))
            most_systematic = systematicity_scores.head(round(len(systematicity_scores)*percentage_list[n]))
            most_systematic_list = most_systematic["word"].tolist()

            filename = "fsc_measures_strongvsweak" + str(percentage_list[n]) + "_" + str(age_bin) + ".csv"

            targets = list(target_vocab)
            final_df = pd.DataFrame(
                data = targets, 
                columns = ["word"]
            )

            #shuffle list to replace the most systematic words with others and do this n times
            n_subsamples = 100
            for j in range(n_subsamples):
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started random shuffle {} of {}...".format(j + 1, n_subsamples)))
                shuffled_list = random.sample(most_systematic_list, len(most_systematic_list))

                def get_random_string(length):
                    # choose from all lowercase letter
                    letters = string.ascii_lowercase
                    return ''.join(random.choice(letters) for z in range(length))

                replacelist = []
                for x in range(len(most_systematic_list)):
                    replacelist.append(get_random_string(10))

                #replacement of systematic words with other systematic words
                with open(reference_space_file, 'r') as file:
                    filedata = file.read()

                for k in range(len(most_systematic_list)):
                    filedata = filedata.replace("\n" + str(most_systematic_list[k]) + " ", "\n" + str(replacelist[k]) + " ")
                    
                for l in range(len(most_systematic_list)):  
                    filedata = filedata.replace("\n" + str(replacelist[l]) + " ", "\n" + str(shuffled_list[l]) + " ")
                    
                new_reference_space_file = "new_reference_space.txt"
                with open(os.path.join(fsc_dir_strongvsweak, new_reference_space_file), 'w') as file:
                    file.write(filedata)  

                #read new randomly shuffled embeddings space
                new_reference_space = SemanticSpace.from_csv(os.path.join(fsc_dir_strongvsweak, new_reference_space_file), prenorm = True)

                #compute fsc values with the shuffle embedding space
                produced_t2OSC_ld_shuffled = levenshtein_fsc(produced_ortho2neighbors_ld, embedding_space, new_reference_space)

                values = []

                for word in targets:
                    values.append(
                        [word, produced_t2OSC_ld_shuffled[word]]
                    )

                produced = "produced_OSC_shuffled" + str(j+1)

                df = pd.DataFrame(
                    data = values,
                    columns = ["word", produced]
                )

                final_df = pd.merge(final_df, df, on = "word")

            final_df.to_csv(os.path.join(fsc_dir_strongvsweak, filename), index = False, sep = ';')

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done with computing for percentage batch {} of {}...".format(n + 1, len(percentage_list))))

    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done with computation for batch {} out of {}. \n".format(
            i+1, len(reference_w2v_filelist
                        )))
        )
            
            



    
