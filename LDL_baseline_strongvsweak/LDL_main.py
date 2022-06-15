import pandas as pd
import numpy as np
import os
from semspaces.space import SemanticSpace
import json
import random
import string
import itertools
from datetime import datetime
from cross_mapping import cross_mapping
from cosine_similarity_linux import compute_cosine_similarity

def write_df(targets, out_path, cossim): # d_target_phon, d_morph # +month of bin, measure column, size of reference vocab, covariates (compute on reference for targets, things like frequency, snd, etc.)

    values = []
    for word in targets:
        values.append(
            [word, cossim[word]] #len(d_target_phon[word]), d_morph[word]
        )


    final_df = pd.DataFrame(
        data=values,
        columns=["word", "cossim"] #"n_phon", "morph"
    )

    final_df.to_csv(out_path, index=False, sep=';')

if __name__ == '__main__':

    
    reference_NDL_space_filebase = "/home/gcassani/systematicity-zeroshot/Reference_NDL_spaces"
    target_NDL_space_filebase = "/home/gcassani/systematicity-zeroshot/Target_NDL_spaces"
    produced_reference_NDL_filebase = "/home/gcassani/systematicity-zeroshot/Produced_reference_NDL_spaces"
    reference_form_spaces_filebase = "/home/gcassani/systematicity-zeroshot/Reference_form_spaces"
    target_form_spaces_filebase = "/home/gcassani/systematicity-zeroshot/Target_form_spaces"
    reference_wordcounts_filebase = "/home/gcassani/systematicity-zeroshot/Wordcounts_reference"
    target_wordcounts_filebase = "/home/gcassani/systematicity-zeroshot/Wordcounts_targets"
    ldl_dir_strongvsweak = "/home/gcassani/systematicity-zeroshot/output_data/ldl/strongvsweak"
    if not os.path.exists(ldl_dir_strongvsweak):
        os.makedirs(ldl_dir_strongvsweak)

    '''
    ldl_dir_strongvsweak = "C:/Users/wgvan/Documents/Universiteit_Nijmegen/jaar2/thesis/Project_Code/LDL_output_dir/random_baselines"
    reference_NDL_space_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/Reference_NDL_spaces"
    target_NDL_space_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/Target_NDL_spaces"
    produced_reference_NDL_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/Produced_reference_NDL_spaces"
    reference_form_spaces_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/FormEmbeddings/Reference_form_spaces"
    target_form_spaces_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/FormEmbeddings/Target_form_spaces"
    reference_wordcounts_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/WordCounts/Wordcounts_reference"
    target_wordcounts_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/WordCounts/Wordcounts_targets"
    '''

    reference_NDL_filelist = sorted(os.listdir(reference_NDL_space_filebase))
    target_NDL_filelist = sorted(os.listdir(target_NDL_space_filebase))
    produced_reference_NDL_filelist = sorted(os.listdir(produced_reference_NDL_filebase))
    reference_form_filelist = sorted(os.listdir(reference_form_spaces_filebase))
    target_form_filelist = sorted(os.listdir(target_form_spaces_filebase))
    reference_wordcounts_filelist = sorted(os.listdir(reference_wordcounts_filebase))
    target_wordcounts_filelist = sorted(os.listdir(target_wordcounts_filebase))


    for i in range(len(target_NDL_filelist)):
        print(
            datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S: Started computation for batch {} out of {}. \n".format(
                    i+1, len(target_NDL_filelist
                             )
                )
            )
        )

        reference_space_file = os.path.join(reference_NDL_space_filebase, reference_NDL_filelist[i])
        reference_space = SemanticSpace.from_csv(
            reference_space_file, prenorm = True
        )
        w2v_reference = reference_space.included_words()

        reference_wordcount = json.load(
            open(os.path.join(reference_wordcounts_filebase, reference_wordcounts_filelist[i]))
        )
        age_bin_reference = reference_wordcount["Age_in_Months"]

        reference_form = SemanticSpace.from_csv(
            os.path.join(reference_form_spaces_filebase, reference_form_filelist[i]), prenorm = True
        )

        reference_vocab = set(w2v_reference)

        print('The target vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(reference_vocab)))

        reference_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in reference_vocab))
        reference_space_array = np.vstack(tuple(reference_space.get_vector(w) for w in reference_vocab))

        #check if the file containing all fsc values already exists and open it, if not, compute fsc values and save this to file
        filename_full = "full_ldl_measures_strongvsweak_" + str(age_bin_reference) + ".csv"
        out_path = os.path.join(ldl_dir_strongvsweak, filename_full)
        try:
            systematicity_file = pd.read_csv(out_path, sep = ";")
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} found and loaded.".format(out_path)))
        except FileNotFoundError:
            print(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing cross-mappings.")
            )
            LDLreference_space = cross_mapping(reference_form_array, reference_space_array, reference_form_array, reference_vocab)
            print(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cross-mappings. \n")
            )

            #compute cosine similarities
            print(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing cosine similarities.")
            )
            cossim = compute_cosine_similarity(LDLreference_space, reference_space, reference_vocab)
            print(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cosine similarities. \n")
            )

            write_df(reference_vocab, out_path, cossim)
            systematicity_file = pd.read_csv(out_path, sep = ";")
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: File {} created.".format(out_path)))

        embedding_space = SemanticSpace.from_csv(
            os.path.join(target_NDL_space_filebase, target_NDL_filelist[i]), prenorm = True
        )
        w2v_words = embedding_space.included_words()

        embedding_form = SemanticSpace.from_csv(
            os.path.join(target_form_spaces_filebase, target_form_filelist[i]), prenorm = True
        )

        wordcount = json.load(open(os.path.join(target_wordcounts_filebase, target_wordcounts_filelist[i])))
        age_bin = wordcount["Age_in_Months"]

        child_produced_space = SemanticSpace.from_csv(
            os.path.join(produced_reference_NDL_filebase, produced_reference_NDL_filelist[i]), prenorm = True
        )
        w2v_child_produced = child_produced_space.included_words()

        systematicity_scores = systematicity_file.sort_values(by=['cossim'], ascending = False)

        percentage_list = [0.1, 0.3, 0.5, 0.7, 0.9]

        # obtain target vocabulary: words which are in the current vocab, but not in the reference one
        # (and occur at least twice)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Creating necessary vocabularies and arrays. \n"))
        precursor_target_vocab = set(w2v_reference.symmetric_difference(w2v_words))
        target_vocab = []
        for word in precursor_target_vocab:
            try:
                if wordcount[word] >= 2:
                    target_vocab.append(word)
            except KeyError:
                continue

        #obtain target form array (needed for the cross-mapping function)
        target_form_array = np.vstack(tuple(embedding_form.get_vector(w) for w in target_vocab))

        # obtain the form embedding arrays for the child-produced reference vocab (needed for the cross-mapping function)
        produced_reference_vocab = list(w2v_reference.intersection(w2v_child_produced))
        produced_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in produced_reference_vocab))

        print('The target vocabulary for retrieving form-based neighbors consists of {} words.'.format(
            len(target_vocab))
        )

        print('The child-produced reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(
            len(produced_reference_vocab))
        )

        for n in range(len(percentage_list)):
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing for percentage batch {} of {}...".format(n + 1, len(percentage_list))))
            most_systematic = systematicity_scores.head(round(len(systematicity_scores)*percentage_list[n]))
            most_systematic_list = most_systematic["word"].tolist()

            filename = "ldl_measures_strongvsweak" + str(percentage_list[n]) + "_" + str(age_bin) + ".csv"

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
                with open(os.path.join(ldl_dir_strongvsweak, new_reference_space_file), 'w') as file:
                    file.write(filedata)  

                #read new randomly shuffled embeddings space
                new_reference_space = SemanticSpace.from_csv(os.path.join(ldl_dir_strongvsweak, new_reference_space_file), prenorm = True)

                #obtain the new reference produced semantic embeddings
                produced_space_array = np.vstack(tuple(new_reference_space.get_vector(w) for w in produced_reference_vocab))
                
                #compute cross-mappings
                produced_LDLtarget_space = cross_mapping(
                    produced_form_array, produced_space_array, target_form_array, target_vocab
                )

                #compute cosine similarities
                produced_cosdist = compute_cosine_similarity(produced_LDLtarget_space, embedding_space, target_vocab)
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cosine."))

                values = []

                for word in targets:
                    values.append(
                        [word, produced_cosdist[word]]
                    )
                
                produced = "produced_LDL_shuffled" + str(j+1)

                df = pd.DataFrame(
                    data = values,
                    columns = ["word", produced]
                )

                final_df = pd.merge(final_df, df, on = "word")

            final_df.to_csv(os.path.join(ldl_dir_strongvsweak, filename), index = False, sep = ';')

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done with computing for percentage batch {} of {}...".format(n + 1, len(percentage_list))))

    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done with computation for batch {} out of {}. \n".format(
            i+1, len(target_NDL_filelist
                        )))
        )

                
        



    


        