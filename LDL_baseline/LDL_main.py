import pandas as pd
import numpy as np
import os
from semspaces.space import SemanticSpace
import json
import random
import copy
from datetime import datetime
from cross_mapping import cross_mapping
from cosine_similarity import compute_cosine_similarity

random_baseline = True

def write_df(targets, out_path, full_cossim, produced_cossim, most_cossim, wordcount, measure, reference_size, embedding_type): # d_target_phon, d_morph # +month of bin, measure column, size of reference vocab, covariates (compute on reference for targets, things like frequency, snd, etc.)

    """
    :param targets:             iterable, containing target words
    :param out_path:            str, indicating the path where to save the combined df
    :param full_cossim:         dict, maps target words to cosine similarity values computed using the true semantic space and the semantic space obtained via LDL on the whole reference vocab (systematicity measure)
    :param produced_cossim:     dict, maps target words to cosine similarity values computed using the true semantic space and the semantic space obtained via LDL on the part of the reference vocab which was also produced by the child (systematicity measure)
    :param most_cossim:         dict, maps target words to cosine similarity values computed using the true semantic space and the semantic space obtained via LDL on the 20% most used words in the reference vocab (systematicity measure)
    :param wordcount            dict, maps target words to their word count and also contains the age bin and if the speech is directed or produced
    :param measure              specifies which method was used to get the systematicity measure
    :param reference_size       specifies how many different words are present in the reference vocab
    :param embedding_type       specifies by which method the embedding space was created
    """
    values = []
    for word in targets:
        if word in wordcount:
            wordcount_word = wordcount[word]
        else:
            wordcount_word = 1
        values.append(
            [measure, word, len(word), wordcount_word, full_cossim[word], produced_cossim[word], most_cossim[word], reference_size, wordcount["Age_in_Months"], wordcount["Directed_or_Produced"], embedding_type] #len(d_target_phon[word]), d_morph[word]
        )


    final_df = pd.DataFrame(
        data=values,
        columns=["measure", "word", "length", "wordcount", "full_cossim", "produced_cossim", "most_cossim", "reference size", "age bin", "directed or produced", "embedding type"] #"n_phon", "morph"
    )

    final_df.to_csv(out_path, index=False, sep=';')

if __name__ == '__main__':

    ldl_dir_rnd = "C:/Users/wgvan/Documents/Universiteit_Nijmegen/jaar2/thesis/Project_Code/LDL_output_dir/random_baselines"
    reference_NDL_spaces_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/Reference_NDL_spaces"
    target_NDL_spaces_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/Target_NDL_spaces"
    produced_reference_NDL_spaces_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/Produced_reference_NDL_spaces"
    reference_form_spaces_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/FormEmbeddings/Reference_form_spaces"
    target_form_spaces_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/FormEmbeddings/Target_form_spaces"
    reference_wordcount_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/WordCounts/Wordcounts_reference"
    target_wordcount_filebase = "D:/UniversiteitNijmegen/Thesis/Test/Processed/WordCounts/Wordcounts_targets"

    '''
    #ask for the path to the map in which all file will be stored
    ldl_dir_rnd = input("Please provide a path to a map where all output will be stored\n") 
    if not os.path.exists(ldl_dir_rnd):
        raise ValueError("This directory does not exist, please provide a valid path!")

    #ask for all files needed for the computations
    reference_NDL_spaces_filebase = input("Please provide the path to the directory where all reference NDL spaces are stored\n") 
    if not os.path.exists(reference_NDL_spaces_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    target_NDL_spaces_filebase = input("Please provide the path to the directory where all target NDL spaces are stored\n") 
    if not os.path.exists(target_NDL_spaces_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    produced_reference_NDL_spaces_filebase = input("Please provide the path to the directory where all produced reference NDL spaces are stored\n") 
    if not os.path.exists(produced_reference_NDL_spaces_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    reference_form_spaces_filebase = input("Please provide the path to the directory where all reference form embeddings are stored\n")
    if not os.path.exists(reference_form_spaces_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    target_form_spaces_filebase = input("Please provide the path to the directory where all target form embeddings are stored\n")
    if not os.path.exists(target_form_spaces_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    reference_wordcount_filebase = input("Please provide the path to the directory where all reference wordcounts are stored\n") 
    if not os.path.exists(reference_wordcount_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")

    target_wordcount_filebase = input("Please provide the path to the directory where all target wordcounts are stored\n") 
    if not os.path.exists(target_wordcount_filebase):
        raise ValueError("This directory does not exist, please provide a valid path!")
    '''

    reference_NDL_filelist = os.listdir(reference_NDL_spaces_filebase)
    target_NDL_filelist = os.listdir(target_NDL_spaces_filebase)
    produced_reference_NDL_filelist = os.listdir(produced_reference_NDL_spaces_filebase)
    reference_form_filelist = os.listdir(reference_form_spaces_filebase)
    target_form_filelist = os.listdir(target_form_spaces_filebase)
    reference_wordcounts_filelist = os.listdir(reference_wordcount_filebase)
    target_wordcounts_filelist = os.listdir(target_wordcount_filebase)

    for i in range(len(reference_NDL_filelist)):
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computation for batch {} out of {}. \n".format(i+1, len(reference_NDL_filelist)))
        )

        embedding_space = SemanticSpace.from_csv(os.path.join(target_NDL_spaces_filebase, target_NDL_filelist[i]), prenorm = True)
        w2v_words = embedding_space.included_words()

        reference_space = SemanticSpace.from_csv(os.path.join(reference_NDL_spaces_filebase, reference_NDL_filelist[i]), prenorm = True)
        w2v_reference = reference_space.included_words()

        child_produced_space = SemanticSpace.from_csv(os.path.join(produced_reference_NDL_spaces_filebase, produced_reference_NDL_filelist[i]), prenorm = True)
        w2v_child_produced = child_produced_space.included_words()

        reference_form = SemanticSpace.from_csv(os.path.join(reference_form_spaces_filebase, reference_form_filelist[i]), prenorm = True)
        embedding_form = SemanticSpace.from_csv(os.path.join(target_form_spaces_filebase, target_form_filelist[i]), prenorm = True)

        wordcount = json.load(open(os.path.join(target_wordcount_filebase, target_wordcounts_filelist[i])))
        reference_wordcount = json.load(open(os.path.join(reference_wordcount_filebase, reference_wordcounts_filelist[i])))
        age_bin = wordcount["Age_in_Months"]



        #obtain target vocabulary: words which are in the current vocab, but not in the reference one (and occur at least twice)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Creating necessary vocabularies and arrays. \n"))
        precursor_target_vocab = set(w2v_reference.symmetric_difference(w2v_words))
        target_vocab = []
        for word in precursor_target_vocab:
            try:
                if wordcount[word] >= 2:
                    target_vocab.append(word)
            except KeyError:
                continue

        #target_vocab = target_vocab[:10]

        #obtain target form array (needed for the cross-mapping function)
        #target_form_array = embedding_form.get_vector(target_vocab[0])
        #for word in target_vocab[1:]:
        #    new_form_vec = embedding_form.get_vector(word)
        #    target_form_array = np.vstack((target_form_array, new_form_vec))
        target_form_array = np.vstack(tuple(embedding_form.get_vector(w) for w in target_vocab))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done for targets."))

        #obtain the embedding space and form embedding arrays for the full reference vocab (needed for the cross-mapping function)
        full_reference_vocab = list(w2v_reference)
        #full_reference_vocab = full_reference_vocab[:1000]
        #full_space_array = reference_space.get_vector(full_reference_vocab[0])
        #full_form_array = reference_form.get_vector(full_reference_vocab[0])

        full_space_array = np.vstack(tuple(reference_space.get_vector(w) for w in full_reference_vocab))
        #full_space_array = np.array(full_space_array, dtype = np.float128)
        full_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in full_reference_vocab))
        #full_form_array = np.array(full_form_array, dtype = np.float128)

        #for word in full_reference_vocab[1:]:
        #    new_space_vec = 
        #    full_space_array = np.vstack((full_space_array, new_space_vec))
        #    new_form_vec = 
        #    full_form_array = np.vstack((full_form_array, new_form_vec))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done for full."))

        #obtain the embedding space and form embedding arrays for the child-produced reference vocab (needed for the cross-mapping function)
        produced_reference_vocab = list(w2v_reference.intersection(w2v_child_produced))
        #produced_reference_vocab = produced_reference_vocab[:1000]
        #produced_space_array = reference_space.get_vector(produced_reference_vocab[0])
        #produced_form_array = reference_form.get_vector(produced_reference_vocab[0])

        produced_space_array = np.vstack(tuple(reference_space.get_vector(w) for w in produced_reference_vocab))
        produced_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in produced_reference_vocab))

        #for word in produced_reference_vocab[1:]:
        #   new_space_vec = reference_space.get_vector(word)
        #   produced_space_array = np.vstack((full_space_array, new_space_vec))
        #   new_form_vec = reference_form.get_vector(word)
        #   produced_form_array = np.vstack((full_form_array, new_form_vec))

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
    
        #most_used_reference_vocab = most_used_reference_vocab[:1000]
        #most_space_array = reference_space.get_vector(most_used_reference_vocab[0])
        #most_form_array = reference_form.get_vector(most_used_reference_vocab[0])

        most_space_array = np.vstack(tuple(reference_space.get_vector(w) for w in most_used_reference_vocab))
        most_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in most_used_reference_vocab))

        #for word in most_used_reference_vocab[1:]:
        #    new_space_vec = reference_space.get_vector(word)
        #    most_space_array = np.vstack((full_space_array, new_space_vec))
        #    new_form_vec = reference_form.get_vector(word)
        #    most_form_array = np.vstack((full_form_array, new_form_vec))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done for most used."))

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done creating necessary vocabularies and arrays. \n"))
        print('The target vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(target_vocab)))
        print('The full reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(full_reference_vocab)))
        print('The child-produced reference vocabulary for retrieving form-based neighbors consists of {} words.'.format(len(produced_reference_vocab)))
        print('The most used reference vocabulary for retrieving form-based neighbors consists of {} words. \n'.format(len(most_used_reference_vocab)))

        #instantiate the random baseline sampling
        n_subsamples = 100
        seeds = random.sample(range(0, 100000000), n_subsamples)
        print(datetime.now().strftime(
            "%d/%m/%Y %H:%M:%S: Started computing cross-mappings and cosine similarities from {} random permutations of the embeddings...".format(n_subsamples)
        ))

        #compute cross-mappings from random permutations of the word embeddings, repeat 1000 times and save measures to file
        random_embeddings = copy.deepcopy(embedding_space)
        
        filename = "ldl_random_baseline" + str(age_bin) + "_4.csv"

        targets = list(target_vocab)
        final_df = pd.DataFrame(
            data = targets,
            columns = ["word"]
        )

        for i, seed in enumerate(seeds):
            np.random.seed(seed)
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started permutation {} of {}...".format(i + 1, n_subsamples)))
            random_embeddings.vectors = np.random.permutation(random_embeddings.vectors)
        
            #compute cross-mappings
            #full_LDLtarget_space = cross_mapping(full_form_array, full_space_array, target_form_array, target_vocab)
            produced_LDLtarget_space = cross_mapping(produced_form_array, produced_space_array, target_form_array, target_vocab)
            #most_LDLtarget_space = cross_mapping(most_form_array, most_space_array, target_form_array, target_vocab)

            #compute cosine similarities
            #full_cossim = compute_cosine_similarity(full_LDLtarget_space, embedding_space, target_vocab)
            produced_cossim = compute_cosine_similarity(produced_LDLtarget_space, embedding_space, target_vocab)
            #most_cossim = compute_cosine_similarity(most_LDLtarget_space, embedding_space, target_vocab)

            values = []

            for word in targets:
                values.append(
                    [word, produced_cossim[word]] #full_cossim[word], most_cossim[word]
                )
            
            #full = "full_LDL_rnd" + str(i+1)
            produced = "produced_LDL_rnd" + str(i+1)
            #most = "most_LDL_rnd" + str(i+1)

            df = pd.DataFrame(
                data = values,
                columns = ["word", produced] #full, most
            )

            final_df = pd.merge(final_df, df, on = "word")
        
        final_df.to_csv(os.path.join(ldl_dir_rnd, filename), index = False, sep = ';')
    
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done with computation for batch {} out of {}. \n".format(i+1, len(reference_NDL_filelist)))
        )
        

        
        



    


        