import pandas as pd
import numpy as np
import os
from semspaces.space import SemanticSpace
import json
from datetime import datetime
from cross_mapping import cross_mapping
from cosine_distance import compute_cosine_distance

def write_df(targets, out_path, produced_cossim, wordcount, reference_size): # d_target_phon, d_morph # +month of bin, measure column, size of reference vocab, covariates (compute on reference for targets, things like frequency, snd, etc.)

    """
    :param targets:             iterable, containing target words
    :param out_path:            str, indicating the path where to save the combined df
    :param produced_cossim:     dict, maps target words to cosine similarity values computed using the true semantic space and the semantic space obtained via LDL on the part of the reference vocab which was also produced by the child (systematicity measure)
    :param wordcount            dict, maps target words to their word count and also contains the age bin
    :param measure              specifies which method was used to get the systematicity measure
    :param reference_size       specifies how many different words are present in the reference vocab
    """
    values = []
    for word in targets:
        if word in wordcount:
            wordcount_word = wordcount[word]
        else:
            wordcount_word = 1
        values.append(
            [word, len(word), wordcount_word, produced_cossim[word], reference_size, wordcount["Age_in_Months"]]
        )


    final_df = pd.DataFrame(
        data=values,
        columns=["word", "length", "wordcount", "produced_cossim", "reference size", "age bin"] 
    )

    final_df.to_csv(out_path, index=False, sep=';')

if __name__ == '__main__':

    ldl_dir = "/home/gcassani/systematicity-zeroshot/output_data/ldl/true_ldl_values/"
    if not os.path.exists(ldl_dir):
        os.makedirs(ldl_dir)
    reference_NDL_spaces_filebase = "/home/gcassani/systematicity-zeroshot/Reference_NDL_spaces"
    target_NDL_spaces_filebase = "/home/gcassani/systematicity-zeroshot/Target_NDL_spaces"
    produced_reference_NDL_spaces_filebase = "/home/gcassani/systematicity-zeroshot/Produced_reference_NDL_spaces"
    reference_form_spaces_filebase = "/home/gcassani/systematicity-zeroshot/Reference_form_spaces"
    target_form_spaces_filebase = "/home/gcassani/systematicity-zeroshot/Target_form_spaces"
    reference_wordcount_filebase = "/home/gcassani/systematicity-zeroshot/Wordcounts_reference"
    target_wordcount_filebase = "/home/gcassani/systematicity-zeroshot/Wordcounts_targets"

    reference_NDL_filelist = sorted(os.listdir(reference_NDL_spaces_filebase))
    target_NDL_filelist = sorted(os.listdir(target_NDL_spaces_filebase))
    produced_reference_NDL_filelist = sorted(os.listdir(produced_reference_NDL_spaces_filebase))
    reference_form_filelist = sorted(os.listdir(reference_form_spaces_filebase))
    target_form_filelist = sorted(os.listdir(target_form_spaces_filebase))
    reference_wordcounts_filelist = sorted(os.listdir(reference_wordcount_filebase))
    target_wordcounts_filelist = sorted(os.listdir(target_wordcount_filebase))

    for i in range(len(reference_NDL_filelist)):
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computation for batch {} out of {}. \n".format(
                i+1, len(reference_NDL_filelist)))
        )

        embedding_space = SemanticSpace.from_csv(
            os.path.join(target_NDL_spaces_filebase, target_NDL_filelist[i]), prenorm = True
        )
        w2v_words = embedding_space.included_words()

        reference_space = SemanticSpace.from_csv(
            os.path.join(reference_NDL_spaces_filebase, reference_NDL_filelist[i]), prenorm = True
        )
        w2v_reference = reference_space.included_words()

        child_produced_space = SemanticSpace.from_csv(
            os.path.join(produced_reference_NDL_spaces_filebase, produced_reference_NDL_filelist[i]), prenorm = True
        )
        w2v_child_produced = child_produced_space.included_words()

        reference_form = SemanticSpace.from_csv(
            os.path.join(reference_form_spaces_filebase, reference_form_filelist[i]), prenorm = True
        )
        embedding_form = SemanticSpace.from_csv(
            os.path.join(target_form_spaces_filebase, target_form_filelist[i]), prenorm = True
        )

        wordcount = json.load(open(os.path.join(target_wordcount_filebase, target_wordcounts_filelist[i])))
        reference_wordcount = json.load(
            open(os.path.join(reference_wordcount_filebase, reference_wordcounts_filelist[i]))
        )
        age_bin = wordcount["Age_in_Months"]

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

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done for targets."))

        # obtain the embedding space and form embedding arrays for the full reference vocab (needed for the
        # cross-mapping function)
        full_reference_vocab = list(w2v_reference)

        full_space_array = np.vstack(tuple(reference_space.get_vector(w) for w in full_reference_vocab))
        full_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in full_reference_vocab))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done for full."))

        # obtain the embedding space and form embedding arrays for the child-produced reference vocab (needed for the
        # cross-mapping function)
        produced_reference_vocab = list(w2v_reference.intersection(w2v_child_produced))
        produced_space_array = np.vstack(tuple(reference_space.get_vector(w) for w in produced_reference_vocab))
        produced_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in produced_reference_vocab))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done for produced."))

        #obtain the embedding space and form embedding arrays for the 20% most frequent words in the reference vocab
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

        most_space_array = np.vstack(tuple(reference_space.get_vector(w) for w in most_used_reference_vocab))
        most_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in most_used_reference_vocab))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done for most used."))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done creating necessary vocabularies and arrays. \n"))
        print('The target vocabulary for retrieving form-based neighbors consists of {} words.'.format(
            len(target_vocab))
        )
        print('The full reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(
            len(full_reference_vocab))
        )
        print(
            'The child-produced reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(
                len(produced_reference_vocab)
            )
        )
        print('The most used reference vocabulary for retrieving form-based neighbors consists of {} words. \n'.format(
            len(most_used_reference_vocab))
        )

        #compute cross-mappings
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing cross-mappings.")
        )
        produced_LDLtarget_space = cross_mapping(
            produced_form_array, produced_space_array, target_form_array, target_vocab
        )
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cross-mappings for all. \n")
        )

        #compute cosine similarities
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing cosine similarities.")
        )
        produced_cossim = compute_cosine_distance(produced_LDLtarget_space, embedding_space, target_vocab)
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cosine similarities. \n")
        )

        #write measures to file for subsequent analysis
        size_of_reference = len(produced_LDLtarget_space)
        filename = "ldl_measures" + str(age_bin) + ".csv"
        write_df(target_vocab, os.path.join(ldl_dir, filename), produced_cossim, wordcount, size_of_reference)

        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done with computation for batch {} out of {}. \n".format(
                i+1, len(reference_NDL_filelist))
            )
        )
        



    


        