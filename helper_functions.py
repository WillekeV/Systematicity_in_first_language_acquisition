import json
import numpy as np
import os
import pandas as pd
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec

#function to make NDL training data, consisting of two lists of the same tokens
def make_NDL_training_data(filename_list):

    #filelist is the list of .json files containing the extracted corpus data from the CHILDES corpus extracted with the corpus_reader code

    total = []

    filelist = sorted(os.listdir(filename_list))

    for i in len(filelist):
        with open(os.path.join(filename_list, filelist[i]), 'r') as file:
            data = json.load(file)

        for j in data[0]:
            total.append(j)

        name = "NDL_file_" + str(i) + ".json"
        json.dump((total, total), open(os.path.join(filename_list, name), 'w'))  
    
    print("done")

#function to make sparse NDL vectors
def make_sparse_NDL_vectors(filename_list, cuename_list):

    filelist = sorted(os.listdir(filename_list))
    cuelist = sorted(os.listdir(cuename_list))

    for i in len(filelist):
        numpy_array = np.load(os.path.join(filename_list, filelist[i]))
        with open(os.path.join(cuename_list, cuelist[i]), 'r') as file:
            cues = json.load(file)

        cueslist = []
        for key in cues.keys():
            cueslist.append(key)

        df = pd.DataFrame(numpy_array, index = cueslist, columns = cueslist)
        variance = df.var()

        sort = sorted(variance)
        threshold = sort[-4001]
        newdf = df.loc[:, df.var() > threshold]

        name = "NLD_vectors_" + str(i) + '.txt'
        with open(os.path.join(filename_list, name), 'w') as f:
            f.write(str(newdf.shape[0]))
            f.write(" ")
            f.write(str(newdf.shape[1]))
            f.write("\n")
            f.close()
        newdf.to_csv(os.path.join(filename_list, name), index = True, header = False, sep = ' ', mode = 'a') 

#function to make trigram word form vectors for use in the LDL algorithm
def make_form_vectors(cuename_list):
    
    cuelist = sorted(os.listdir(cuename_list))

    #first create a list of all threegrams available in the corpus
    with open(os.path.join(cuename_list, cuelist[-1]), 'r') as f:
        complete_cues = json.load(f)

    complete_wordlist = []
    for key in complete_cues.keys():
        complete_wordlist.append(key)

    complete_newlist = []
    for word in complete_wordlist:
        word = '#' + word + '#'
        if word == '##':
            word = '# #'
        complete_newlist.append(word)
        
    #create a list of all threegrams
    threegrams = []
    k = 3
    for word in complete_newlist:
        for start in range(len(word)):
            temp = word[start: start + k]
            if len(temp) == k:
                if temp not in threegrams:
                    threegrams.append(temp)

    #for each part of the corpus, make formvectors specifying which threegrams are present in each word
    for i in len(cuelist):
        with open(os.path.join(cuename_list, cuelist[i]), 'r') as file:
            cues = json.load(file)

        wordlist = []
        for key in cues.keys():
            wordlist.append(key)

        #append a # as start and end of word (as in Baayen)
        newlist = []
        for word in wordlist:
            word = '#' + word + '#'
            if word == '##':
                word = '# #'
            newlist.append(word)

        #create dataframe
        d = pd.DataFrame(0, index=newlist, columns=threegrams)

        #check if 3gram is in word, if yes, the corresponding cel will have 1 added to it
        for word in newlist:
            for ngram in threegrams:
                if ngram in word:
                    d.loc[word, ngram] += 1
                    
        d.set_axis(wordlist, axis='index', inplace = True)

        name = "Formvector" + str(i) + ".txt"
        with open(os.path.join(cuename_list, name), 'w') as f:
            f.write(str(d.shape[0]))
            f.write(" ")
            f.write(str(d.shape[1]))
            f.write('\n')
            f.close()
        d.to_csv(os.path.join(cuename_list, name), index = True, header = False, sep = ' ', mode = 'a')

    print("done")

#function to make fasttext vectors
def make_fasttext_vectors(filename_list):

    filelist = sorted(os.listdir(filename_list))

    #first iteration of the model
    for i in len(filelist[:1]):
        with open(os.path.join(filename_list, filelist[i]), 'r') as file:
            data = json.load(file)
        
        model = FastText(data[0], size = 100, window = 10, min_count = 1, sg = 1, iter = 20)
        model_name = "FT_model_" + str(i) + ".model"
        vector_name = "FT_vectors_" + str(i) + ".txt"
        model.save(os.path.join(filename_list, model_name))
        model.wv.save_word2vec_format(os.path.join(filename_list, vector_name), binary = False)

    #later iterations of the model: expand upon the first
    for i in len(filelist[1:]):
        with open(os.path.join(filename_list, filelist[i]), 'r') as file:
            data = json.load(file)
        previous_model = "FT_model_" + str(i-1) + ".model"
        model = FastText.load(os.path.join(filename_list, previous_model))
        model.build_vocab(data[0], update = True)
        model.train(data[0], total_examples = len(data[0]), epochs=model.epochs)
        model_name = "FT_model_" + str(i) + ".model"
        vector_name = "FT_vectors_" + str(i) + ".txt"
        model.save(os.path.join(filename_list, model_name))
        model.wv.save_word2vec_format(os.path.join(filename_list, vector_name), binary = False)

    print("done")

#function to make word2vec vectors
def make_word2vec_vectors(filename_list):

    filelist = sorted(os.listdir(filename_list))

    #first iteration of the model
    for i in len(filelist[:1]):
        with open(os.path.join(filename_list, filelist[i]), 'r') as file:
            data = json.load(file)
        
        model = Word2Vec(data[0], size = 100, window = 10, min_count = 1, sg = 1, iter = 20)
        model_name = "w2v_model_" + str(i) + ".model"
        vector_name = "w2v_vectors_" + str(i) + ".txt"
        model.save(os.path.join(filename_list, model_name))
        model.wv.save_word2vec_format(os.path.join(filename_list, vector_name), binary = False)

    #later iterations of the model: expand upon the first
    for i in len(filelist[1:]):
        with open(os.path.join(filename_list, filelist[i]), 'r') as file:
            data = json.load(file)
        previous_model = "w2v_model_" + str(i-1) + ".model"
        model = Word2Vec.load(os.path.join(filename_list, previous_model))
        model.build_vocab(data[0], update = True)
        model.train(data[0], total_examples = len(data[0]), epochs=model.epochs)
        model_name = "w2v_model_" + str(i) + ".model"
        vector_name = "w2v_vectors_" + str(i) + ".txt"
        model.save(os.path.join(filename_list, model_name))
        model.wv.save_word2vec_format(os.path.join(filename_list, vector_name), binary = False)

    print("done")

#function to obtain wordcounts for each word in each part of the corpus and the child's age
def wordcounter(filename_list):

    filelist = sorted(os.listdir(filename_list))

    for i in len(filelist):
        with open(os.path.join(filename_list, filelist[i]), 'r') as file:
            data = json.load(file)

        countDict = {}
        for sentence in data[0]:
            for word in sentence:
                if word in countDict:
                    countDict[word] += 1
                else:
                    countDict[word] = 1
        
        countDict["Age_in_Months"] = 24
        countDict["Directed_or_Produced"] = "Directed"

        name = "wordcount" + str(i) + ".json"
        json.dump(countDict, open(os.path.join(filename_list, name), 'w'))

#function to obtain AoA mappings to see in which age_bin the child uttered the word for the first time
def AoA_mapping(wordcountname_list):

    wordcount_list = sorted(os.listdir(wordcountname_list))

    child_produced_AoA = {}

    for i in len(wordcount_list):
        with open(os.path.join(wordcountname_list, wordcount_list[i]), 'r') as file:
            wordcount = json.load(file)

        age_bin = wordcount["Age_in_Months"]

        wordcount.pop("Directed_or_Produced")
        wordcount.pop("Age_in_Months")

        for word in wordcount:
            if word in child_produced_AoA:
                if child_produced_AoA[word] == "later":
                    child_produced_AoA[word] = age_bin
                else:
                    continue
            else:
                if int(wordcount[word]) > 1:
                    child_produced_AoA[word] = "30"
                else:
                    child_produced_AoA[word] = "later"

        name = "AoAProduced.json"
        json.dump(child_produced_AoA, open(os.path.join(wordcountname_list, name), 'w'))




if __name__ == '__main__':  

    NDL_filelist = '' #input the name of the directory you store all your corpus extracted files in
    make_NDL_training_data(NDL_filelist)

    NDL_matrix_list = '' #input the name of the directory you store all your NDL association matrices in, which were obtained with running the ndl.py file from NDL_semantic_vectors
    NDL_cues_list = '' #input the name of the directory you store all your NDL cueIDs in, which were obtained with running the ndl.py file from NDL_semantic_vectors
    make_sparse_NDL_vectors(NDL_matrix_list, NDL_cues_list)
    make_form_vectors(NDL_cues_list)

    WordVector_filelist = '' #input the name of the directory you store all your corpus extracted files in
    make_fasttext_vectors(WordVector_filelist)
    make_word2vec_vectors(WordVector_filelist)
    wordcounter(WordVector_filelist)


    