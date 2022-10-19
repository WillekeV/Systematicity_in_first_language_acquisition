import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import panphon
import epitran
import re
import pickle

#the semantic to vision model

#definition of the semantic model
def get_sem_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(50, input_dim=n_inputs, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs))
    model.compile(loss='cosine_similarity', optimizer=keras.optimizers.Adam(lr=0.001))
    return model

#definition of the evaluation of the first timestep of the model, gives as result the cosine_loss and the predicted vision vectors for the test set
#needed as parameters: The semantic vectors as input and the vision vectors as targets and if the model should be randomized or not (for random baseline)
def evaluate_sem_model_first_timestep(X, y, random_state):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # prepare data
    X_train = X[:511]
    X_test = X[511:]
    y_train = y[:511]
    y_test = y[511:]
    if random_state:
        y_train = shuffle(y_train, random_state=None)
    else:
        y_train = y_train
    # define model
    model = get_sem_model(n_inputs, n_outputs)
    # fit model
    model.fit(X_train, y_train, verbose=0, epochs=20)
    # evaluate model on test set
    cossim = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)
    # store result
    results.append(cossim)
    return results, predictions

#definition of the evaluation for all other timesteps of the model, gives as result the cosine_loss and the predicted vision vectors for the test set
#needed as parameters: The semantic vectors as training-input, the vision vectors as training-targets, the semantic vectors as test-input and the vision vectors as test-targets
def evaluate_sem_model_other_timesteps(X, y, X2, y2):
    results = list()
    n_inputs, n_outputs = X2.shape[1], y2.shape[1]
    # prepare data
    X_train, X_test = X, X2
    y_train, y_test = y, y2
    # define model
    model = get_sem_model(n_inputs, n_outputs)
    # fit model
    model.fit(X_train, y_train, verbose=0, epochs=20)
    # evaluate model on test set
    cossim = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)
    # store result
    #print('>%.3f' % cossim)
    results.append(cossim)
    return results, predictions


#function to make phonetics vectors for all words
def phon_vectorizer(data, lang):
    epi = epitran.Epitran(dict_ipa.get(lang))
    phon = []
    for word in data:
            ipa_word = epi.transliterate(word)
            phon_vec = ft.word_array(phon_features, ipa_word)
            phon.append(phon_vec)
    phon = keras.preprocessing.sequence.pad_sequences(np.array(phon), padding='post', maxlen=15) # set maxlen
    print("(samples, timesteps, features) =", phon.shape) 
    return phon


#The phonetics to vision model

#definition of the phonetic model
def get_phon_model(input_shape1, input_shape2, n_outputs):
    model=keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(input_shape1, input_shape2)))
    model.add(keras.layers.GRU(units=50, batch_input_shape=(None, input_shape1, input_shape2), return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(keras.layers.Dense(n_outputs))
    model.add(Activation('linear'))
    model.compile(loss='cosine_similarity', optimizer='adam')
    return model

#definition of the evaluation for all timesteps of the phonetic model, gives as result the cosine_loss and the predicted vision vectors for the test set
#needed as parameters: the vision vectors as training-targets, the vision vectors as test-targets, the path to the training dict and the path to the training and test dict
def evaluate_phon_model(ytrain, ytest, path_train, path_test):
    results = list()
    
    with open(path_train, 'rb') as handle:
        train_dict = pickle.load(handle)
    with open(path_test, 'rb') as handle:
        test_dict = pickle.load(handle)
    
    xtrain = train_dict["english"]
    xtest = test_dict["english"]
    
    input_shape1 = xtrain.shape[1]
    input_shape2 = xtrain.shape[2]
    n_outputs = ytrain.shape[1]
    
    model = get_phon_model(input_shape1, input_shape2, n_outputs)
    
    model.fit(xtrain, ytrain, epochs = 20)
    cossim = model.evaluate(xtest, ytest)
    predictions = model.predict(xtest)
    
    results.append(cossim)
    return results, predictions


#The semantics and phonetics to vision model

#definition of the semantic + phonetic model
def get_semphon_model(n_inputs, n_outputs, input_shape1, input_shape2):
    sem_model = Sequential()
    sem_model.add(Dense(50, input_dim=n_inputs, activation='relu'))
    sem_model.add(Dropout(0.2))
    sem_model.add(Dense(n_outputs))

    phon_model=keras.models.Sequential()
    phon_model.add(keras.layers.Masking(mask_value=0., input_shape=(input_shape1, input_shape2)))
    phon_model.add(keras.layers.GRU(units=50, batch_input_shape=(None, input_shape1, input_shape2), return_sequences=False))
    phon_model.add(Activation('relu'))
    phon_model.add(Dropout(0.2))
    phon_model.add(keras.layers.Dense(n_outputs))
    phon_model.add(Activation('linear'))

    merged_layers = layers.concatenate([sem_model.output, phon_model.output])
    x = Dense(50, activation='relu')(merged_layers)
    x = Dropout(0.2)(x)
    x = Dense(n_outputs)(x)
    merged_model = Model([sem_model.input, phon_model.input], [x])
    merged_model.compile(loss='cosine_similarity', optimizer='adam')
    return merged_model

#definition of the evaluation for all timesteps of the semantic + phonetic model, gives as result the cosine_loss and the predicted vision vectors for the test set
#needed as parameters: the semantic vectors as training-input, the vision vectors as training-targets, the semantic vectors as test-input, the vision vectors as test-targets and the path to the training and test dicts
def evaluate_semphon_model(sem_vectors_train, vis_vectors_train, sem_vectors_test, vis_vectors_test, path_train, path_test):
    results = list()
    n_inputs, n_outputs = sem_vectors_test.shape[1], vis_vectors_test.shape[1]
    
    with open(path_train, 'rb') as handle:
        train_dict = pickle.load(handle)
    with open(path_test, 'rb') as handle:
        test_dict = pickle.load(handle)
    
    phon_vectors_train = train_dict["english"]
    phon_vectors_test = test_dict["english"]
    
    input_shape1 = phon_vectors_train.shape[1]
    input_shape2 = phon_vectors_train.shape[2]
    
    # define model
    model = get_semphon_model(n_inputs, n_outputs, input_shape1, input_shape2)
    # fit model
    model.fit([sem_vectors_train, phon_vectors_train], vis_vectors_train, epochs=20)
    # evaluate model on test set
    cossim = model.evaluate([sem_vectors_test, phon_vectors_test], vis_vectors_test)
    predictions = model.predict([sem_vectors_test, phon_vectors_test])
    # store result
    results.append(cossim)
    return results, predictions


#function to calculate cosine similarity and word-ranking
#needs as input the true vision vectors and the predicted vision vectors as arrays (with the vectors being alligned)
def calculate_cossim_and_ranking(true_vis_vectors, predicted_vis_vectors):
    cossim_list = []
    for i in range(len(true_vis_vectors)):
        cossim = cosine_similarity(true_vis_vectors[i].reshape(1, -1), predicted_vis_vectors[i].reshape(1, -1))
        cossim_list.append(cossim[0][0])

    ranking_list = []
    for k in range(len(predicted_vis_vectors)):
        cossim_list_word = []
        for j in range(len(true_vis_vectors)):
            cossim = cosine_similarity(true_vis_vectors[k].reshape(1, -1), predicted_vis_vectors[j].reshape(1, -1))
            cossim_list_word.append(cossim[0][0])

        to_find = cosine_similarity(true_vis_vectors[k].reshape(1, -1), predicted_vis_vectors[k].reshape(1, -1))
        sorted_list = sorted(cossim_list_word, reverse = True)
        index = sorted_list.index(to_find)
        ranking_list.append(index)

    return(cossim_list, ranking_list)

if __name__ == '__main__':

    #call the semantic model:
    semantic_vectors_timestep1 = None #specify your file containing semantic vectors for the first timestep here
    vision_vectors_timestep1 = None #specify your file containing vision vectors for the first timestep here
    random_baseline = False #specifies if you want to run the random baseline or not
    results_timestep1, predictions_timestep1 = evaluate_sem_model_first_timestep(semantic_vectors_timestep1, vision_vectors_timestep1, random_baseline)

    semantic_vectors_timestep2 = None #specify your file containing semantic vectors for the second timestep here
    vision_vectors_timestep2 = None #specify your file containing vision vectors for the second timestep here
    #call the semantic models for the next timesteps, you can repeat this process for timesteps 3, 4, etc. always using the data from the previous timestep as training data
    #If you want to run a random baseline for these timesteps, you need to randomize your semantic or vision vectors beforehand
    results_timestep2, predictions_timestep2 = evaluate_sem_model_other_timesteps(semantic_vectors_timestep1, vision_vectors_timestep1, semantic_vectors_timestep2, vision_vectors_timestep2)


    #for the phonetics to vision model, in order to create the phonetic vectors:
    #open file with language information and make a dictionary mapping the languages to their epitran abbreviation to be used in the epitran model
    f = open("languages_epi&fasttext", "r").read().split('\n')
    dict_ipa = {}
    for item in f:
        a = re.sub(r"([A-Z][a-z]+)([ \t]+)([a-z]+\-[A-Z][a-z]+)", r"\1", item)
        b = re.sub(r"([A-Z][a-z]+)([ \t]+)([a-z]+\-[A-Z][a-z]+)", r"\3", item)
        dict_ipa.update( {a : b} )

    #specifying the panphon features to be used by the epitran/panphon model    
    ft = panphon.FeatureTable()
    phon_features = ['syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas', 'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi', 'lo', 'back', 'round', 'velaric', 'tense', 'long']

    #make phonetic vectors to run the phonetics to vision NN
    training_wordlist = None #specify the file that contains all training words here
    training_phonetic_vectors = phon_vectorizer(training_wordlist, "English")
    train_dict = {"english" : training_phonetic_vectors}
    filepath_train = None #specify where you want to save your train_dict here
    with open(filepath_train, 'wb') as f:
        pickle.dump(train_dict, f)
    test_wordlist = None #specify the file that contains all test words here
    test_phonetic_vectors = phon_vectorizer(test_wordlist, "English")
    test_dict = {"english" : test_phonetic_vectors}
    filepath_test = None #specify where you want to save your test_dict here
    with open(filepath_test, 'wb') as f:
        pickle.dump(test_dict, f)

    #call the phonetic model:
    vision_vectors_train = None #specify your file containing vision vectors for the training-set here
    vision_vectors_test = None #specify your file containing vision vectors for the testing-set here
    #you can run this model for all timesteps. If you want to run a random baseline, you need to randomize your vectors beforehand
    phon_results, phon_predictions = evaluate_phon_model(vision_vectors_train, vision_vectors_test, filepath_train, filepath_test)

    #call the semantic + phonetic model:
    sem_vectors_train = None #specify your file containing semantic vectors for the training-set here
    sem_vectors_test = None #specify your file containing semantic vectors for the testing-set here
    vision_vectors_train = None #specify your file containing vision vectors for the training-set here
    vision_vectors_test = None #specify your file containing vision vectors for the testing-set here
    #you can run this model for all timesteps. If you want to run a random baseline, you need to randomize your vectors beforehand
    semphon_results, semphon_predictions = evaluate_semphon_model(sem_vectors_train, vision_vectors_train, sem_vectors_test, vision_vectors_test, filepath_train, filepath_test)

    #call the function to calculate the cosine similarities of each word between the predicted and true vision vectors and to calculate the ranking of each predicted vector
    #explanation of ranking: if the ranking of a predicted word to it's true word is 5, there are 4 other true word vectors closer to the predicted vector compared to the desired one
    predictions = None #specify either your file containing the predicted vectors here or directly specify the predictions from an NN here
    true = None #specify the true vision vectors here (as an array), they need to be alligned to your predicted vectors
    #you can call this function for any pair of predictions and true vectors (as long as they are alligned), you can then save them in a file however you wish
    cossims, rankings = calculate_cossim_and_ranking(true, predictions)