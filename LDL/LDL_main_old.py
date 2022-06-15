import pandas as pd
import numpy as np
import os
from semspaces.space import SemanticSpace
import json
from datetime import datetime
from cross_mapping import cross_mapping
from cosine_similarity import compute_cosine_similarity

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

    #ask for the target embedding space file. First line of the file should be the size (for example 10000 300).
    embedding_space_file = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/NDLVectorChildDirected42.txt" #input("Please provide the path file in which the target embedding space is stored.\n") #D:/UniversiteitNijmegen/Thesis/Test/Processed/Word2Vec/W2VChildDirected30.txt
    if not os.path.exists(embedding_space_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the reference embedding space file. First line of the file should be the size (for example 10000 300).
    reference_space_file = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/NDLVectorChildDirected36.txt" #input("Please provide the path file in which the reference embedding space is stored.\n") #D:/UniversiteitNijmegen/Thesis/Test/Processed/Word2Vec/W2VChildDirected24.txt
    if not os.path.exists(reference_space_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for a file in which all form embeddings of the target words are stored
    embedding_form_file = "D:/UniversiteitNijmegen/Thesis/Test/Processed/FormEmbeddings/FormVectorChildDirected42.txt" #input("Please provide the path file in which the form embeddings are stored.\n")
    if not os.path.exists(embedding_form_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for a file in which all form embeddings of the reference words are stored
    reference_form_file = "D:/UniversiteitNijmegen/Thesis/Test/Processed/FormEmbeddings/FormVectorChildDirected36.txt" #input("Please provide the path file in which the form embeddings for the reference vocab are stored.\n")
    if not os.path.exists(reference_form_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the child-produced reference embedding space file. First line of the file should be the size (for example 10000 300).
    child_produced_space_file = "D:/UniversiteitNijmegen/Thesis/Test/Processed/NDL/NDLVectorChildProduced36.txt" #input("Please provide the path file in which the reference child-produced embedding space is stored.\n")
    if not os.path.exists(child_produced_space_file):
        raise ValueError("The file does not exist, please provide a valid file!")

    #ask for the file containing a dictionary of wordcounts for all words in the target vocab. It should also include the age of the child and if the speech is directed or produced.
    wordcount_file = "D:/UniversiteitNijmegen/Thesis/Test/Processed/WordCounts/WordCountChildDirected42.json" #input("Please provide the path to the file containing wordcounts for all words in the target embedding space. \n")
    if not os.path.exists(wordcount_file):
        raise ValueError("This directory does not exist, please provide a valid path!") 

    #ask for the file containing a dictionary of wordscounts for all the words in the reference vocab.
    wordcount_file_reference = "D:/UniversiteitNijmegen/Thesis/Test/Processed/WordCounts/WordCountChildDirected36.json" #input("Please provide the path to the file containing wordcounts for all words in the reference embedding space. \n")
    if not os.path.exists(wordcount_file_reference):
        raise ValueError("This directory does not exist, please provide a valid path!")

    #ask for the path to the map in which all file will be stored
    ldl_dir = "C:/Users/wgvan/Documents/Universiteit_Nijmegen/jaar2/thesis/Project_Code/LDL_output_dir" #input("Please provide a path to a map where all output will be stored")
    if not os.path.exists(ldl_dir):
        raise ValueError("This directory does not exist, please provide a valid path!")

    #loading all data
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started loading the data... \n"))

    embedding_space = SemanticSpace.from_csv(embedding_space_file, prenorm=True)
    w2v_words = embedding_space.included_words()

    reference_space = SemanticSpace.from_csv(reference_space_file, prenorm = True)
    w2v_reference = reference_space.included_words()

    embedding_form = SemanticSpace.from_csv(embedding_form_file, prenorm=True)
    reference_form = SemanticSpace.from_csv(reference_form_file, prenorm=True)

    child_produced_space = SemanticSpace.from_csv(child_produced_space_file, prenorm = True)
    w2v_child_produced = child_produced_space.included_words()

    wordcount = json.load(open(wordcount_file))
    reference_wordcount = json.load(open(wordcount_file_reference))
    age_bin = wordcount["Age_in_Months"]

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done loading data. \n"))

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
    print(full_space_array.dtype)
    #full_space_array = np.array(full_space_array, dtype = np.float128)
    full_form_array = np.vstack(tuple(reference_form.get_vector(w) for w in full_reference_vocab))
    print(full_form_array.dtype)
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

    #compute cross-mappings
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing cross-mappings.")
    )
    #full_LDLtarget_space = cross_mapping(full_form_array, full_space_array, target_form_array, target_vocab)
    #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing for full."))
    produced_LDLtarget_space = cross_mapping(produced_form_array, produced_space_array, target_form_array, target_vocab)
    #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing for produced."))
    #most_LDLtarget_space = cross_mapping(most_form_array, most_space_array, target_form_array, target_vocab)
    #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing for most-used."))
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cross-mappings for all. \n")
    )

    #compute cosine similarities
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing cosine similarities.")
    )
    #full_cossim = compute_cosine_similarity(full_LDLtarget_space, embedding_space, target_vocab)
    #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cosine similarities for full."))
    produced_cossim = compute_cosine_similarity(produced_LDLtarget_space, embedding_space, target_vocab)
    #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cosine similarities for child-produced."))
    #most_cossim = compute_cosine_similarity(most_LDLtarget_space, embedding_space, target_vocab)
    #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cosine similarities for most-used."))
    print(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done computing cosine similarities. \n")
    )

    #write measures to file for subsequent analysis
    size_of_reference = len(produced_LDLtarget_space)
    filename = "ldl_measures" + str(age_bin) + ".csv"
    write_df(target_vocab, os.path.join(ldl_dir, filename), produced_cossim, wordcount, size_of_reference)
        



    


        