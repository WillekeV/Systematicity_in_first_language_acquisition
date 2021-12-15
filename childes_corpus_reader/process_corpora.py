import glob
import os
import json
from CorpusReader import ModifiedCHILDESCorpusReader


# function to handle clitics
def handle_clitics(token_sentence, full_forms=False):

    """
    replaces clitics like "'m" or "n't" with their full orthographic forms ("'m" --> "am", "n't" --> "not")
    """

    s = ' '.join(token_sentence)
    s = ' ' + s + ' '

    if full_forms:
        # map clitics to their full forms
        clitic_dict = {"'m ": ' am ', "'re ": ' are ', "'s ": ' is ', "'ve ": ' have ', "n't ": ' not ',
                       "'ll": ' will ', "'d ": ' would ', 'not ': ' not '}
    else:
        # map clitics to themselves, prepending a space for segmentation
        clitic_dict = {"'m ": " 'm ", "'re ": " 're ", "'s ": " 's ", "'ve ": " 've ", "n't ": ' not ',
                       "'ll": " 'll ", "'d ": " 'd ", 'not ': ' not '}

    # replace clitcs with full form
    for clitic, full_form in clitic_dict.items():
        s = s.replace(clitic, full_form)

    return s.split()

# function to get the lemma/lexeme (naming convention) for each word
def get_lexeme(stem, pos_tag):

    # remove regular and irregular morphology tags (these are part of the stem string)
    stem = stem.split('&')[0].split('-')[0]

    # a lexeme a stem plus pos tag (e.g. 'bike~n' for the noun 'bike')
    lexeme = stem + '~' + pos_tag

    return lexeme

# function to get lemmas/lexemes from the sentences
def get_lexeme_sent(stem_sent):

    lexeme_sent = []
    for idx, word_postag in enumerate(stem_sent):
        word, pos_tag = word_postag

        # replace 'None' pos tag with string for easier processing
        # ('None' is used with unintelligible words)
        if not pos_tag:
            pos_tag = 'None'

        word = word.lower()
        pos_tag = pos_tag.lower()

        # a lexeme is a stem plus pos tag (e.g. 'bike#n' for the noun 'bike')
        lexeme_sent.append(get_lexeme(word, pos_tag))

    return lexeme_sent

def get_tokens_lexemes_from_corpus(corpus, root_folder):

    """
    param corpus: the current corpus to be considered (one .xml file from CHILDES)
    this function returns multiple lists: 
        a list of tokens uttered by anyone but the target child
        a list of corresponding lexemes and PoS tags of the above mentioned tokens
        a list of tokens uttered by the target child(ren)
        a list of corresponding lexemes and PoS tags of the above mentioned tokens
        a list of ages of the target child(ren) (most of the time only 1, but could be multiple)
        a list with the name of the corpora that where processed (in this case only one, but the function could be modified to consider more corpora at once)
        a list of all the child names in the corpus
    """
    
    lexeme_transcripts = []     # lexemes from non-target children (a lexeme is a stem together with its POS tag)
    token_transcripts = []      # tokens corresponding to lexemes from non-target children
    child_lexeme_transcripts = []    # lexemes from target children
    child_token_transcripts = []    # tokens corresponding to lexemes from target children

    ages = []    # list of ages
    corpora = []    #list of corpora
    child_names = []    #list of child names

    token_counter = 0    #token counter to check how many tokens are in one particular corpus (use if needed for checking)
    
    corpus_reader = ModifiedCHILDESCorpusReader(str(root_folder), corpus) # specify the root folder here
    corpus_participants = corpus_reader.participants(corpus_reader.fileids())

    childlist = set()    #list of all participant keys from target children
    adultlist = set()    #list of all participant keys from non-target children
        
    age = corpus_reader.age(month=True)
    
    # checks if the age of the children is specified (sometimes this is not the case), if not, the file cannot be used
    # then adds target children keys to the childlist and the rest to the adultlist
    if age:
        for this_corpus_participants in corpus_participants:
            for key in sorted(this_corpus_participants.keys()):
                dct = this_corpus_participants[key]
                for k in sorted(dct.keys()):
                    if k == 'role' and dct[k] == 'Target_Child':
                        childlist.add(key)
                    elif k == 'role' and dct[k] != 'Target_Child':
                        adultlist.add(key)

        # we need the corpus both in stemmed format and in token format
        # stem=True return a list of stem sentences; each stem is given in a tuple with its POS tag, e.g. (car, 'n')
        stem_sents = corpus_reader.tagged_sents(corpus, adultlist, stem=True, relation=False,
                                                strip_space=True, replace=True)
        child_stems = corpus_reader.tagged_sents(corpus, childlist, stem=True, relation=False,
                                                 strip_space=True, replace=True)
        
        # stem=False returns a list of token sentences; tokens are given as plain strings (without POS tags)
        token_sents = corpus_reader.sents(corpus, adultlist, stem=False, relation=False,
                                          strip_space=True, replace=True)        
        child_tokens = corpus_reader.sents(corpus, childlist, stem=False, relation=False,
                                          strip_space=True, replace=True)

        # give a name to the current corpus (is for checking which corpus is done being processed during code execution)
        participant_dict = corpus_reader.participants(corpus)[0]['CHI']
        name = participant_dict['name']
        if len(name) == 0:
            name = str(corpus)

        transcript_tokens = []
        transcript_lexemes = []
        transcript_tokens_child = []
        transcript_lexemes_child = []

        # go through each sentence
        for stems, tokens in zip(stem_sents, token_sents):

            # if the sentence of stems differs in length from the sentence of tokens, that's most likely
            # because clitics are merged with their tokens, whereas in the stemmed format clitics are their
            # own lexical items (= they are given in full form, i.e. "I am" instead of "I'm"
            if len(stems) != len(tokens):
                # remove clitics from tokens and insert them as separate words
                tokens = handle_clitics(tokens)

            # if the lengths still differ, there is a discrepancy between the two coding schemes
            # this is rare but it does happen, if it does: ignore the sentence
            if len(tokens) != len(stems):
                    continue

            # lower-case tokens
            tokens = [t.lower() for t in tokens]

            # extract lexemes from the stemmed sentence (lexeme = stem + POS tag)
            lexemes = get_lexeme_sent(stems)

            transcript_lexemes.append(lexemes)
            transcript_tokens.append(tokens)

            token_counter += len(tokens)
        
        # same for the child produced tokens
        for stems, tokens in zip(child_stems, child_tokens):
            if len(stems) != len(tokens):
                tokens = handle_clitics(tokens)
            if len(tokens) != len(stems):
                    continue

            tokens = [t.lower() for t in tokens]
            lexemes = get_lexeme_sent(stems)
            transcript_lexemes_child.append(lexemes)
            transcript_tokens_child.append(tokens)

        token_transcripts.append(transcript_tokens)
        lexeme_transcripts.append(transcript_lexemes)
        child_lexeme_transcripts.append(transcript_lexemes_child)
        child_token_transcripts.append(transcript_tokens_child)

        ages.append(list(age))    # this needs to be explicitely made into a list, otherwise Python produces a LazyMap for some reason
        corpora.append(str(corpus))
        child_names.append(name)

    print('Processed corpus %s (%s tokens).' % (str(corpus), token_counter)) # prints to check how far in the executing we are
    return token_transcripts, lexeme_transcripts, child_token_transcripts, child_lexeme_transcripts, ages, corpora, child_names

def get_tokens_lexemes(root_folder, corpora):

    """
    param corpora: list of corpus names to be considered (directories in the root folder)
    this function returns multiple lists for all corpora: 
        a list of tokens uttered by anyone but the target child
        a list of corresponding lexemes and PoS tags of the above mentioned tokens
        a list of tokens uttered by the target child(ren)
        a list of corresponding lexemes and PoS tagd of the above mentioned tokens
        a list of ages of the target child(ren) (most of the time only 1, but could be multiple)
        a list with the name of the corpora that where processed (in this case only one, but the function could be modified to consider more corpora at once)
        a list of all the child names in the corpus
    """
    
    n_transcripts = 0

    ret = {'token_transcripts': [],
           'lexeme_transcripts': [],
           'child_token_transcripts': [],
           'child_lexeme_transcripts': [],
           'corpus_names': [],
           'child_ages': [],
           'child_names': []}

    for corpus in corpora:
        print(corpus)
        token_transcripts, lexeme_transcripts, child_token_transcripts, child_lexeme_transcripts, ages, corpus_names, child_names = \
            get_tokens_lexemes_from_corpus(corpus, root_folder)
        ret['token_transcripts'] += token_transcripts
        ret['lexeme_transcripts'] += lexeme_transcripts
        ret['child_token_transcripts'] += child_token_transcripts
        ret['child_lexeme_transcripts'] += child_lexeme_transcripts
        ret['child_ages'] += ages
        ret['corpus_names'] += corpus_names
        ret['child_names'] += child_names

        if len(token_transcripts) == 0:
            print('Corpus is not marked for age and / or child name: %s' % corpus)
            continue

        n_transcripts += len(token_transcripts)

    print('Done processing corpus/corpora')
    print('Nr transcripts: %s' % n_transcripts, len(ret['token_transcripts']))

    return ret

def process_and_save_corpus_data(filelist2, file_dir, root_folder):

    """
    pre-process corpora and pickle result to file directory
    """

    corpora = filelist2   # the filelist that was created before, containing all .xml files

    corpus_dict = get_tokens_lexemes(root_folder, corpora=corpora)
    json.dump(corpus_dict, open(file_dir + '/corpus_EN_child_raw.json', 'w'))

def save_corpus_child_directed(file_dir, age_list, corpus):
    
    """
    This is a function to extract 2 lists from the .json file which was generated with process_and_save_corpus_data()
    The first list is a list of all tokens directed at target children, the second list their corresponding lexemes
    These lists are dumped in .json files, according to the age of the target child
    This function can be modified to dump different parts of the .json file which was generated with process_and_save_corpus_data()
    and also on which basis they need to be in different files (or all in the same file)
    """
    
    # opens the previous .json file and zips the necessary parts of it in zipped
    corpus_dir = file_dir + '/%s_raw.json' % corpus
    cor = json.load(open(corpus_dir, 'r'))
    zipped = list(zip(cor['child_ages'], cor['token_transcripts'], cor['lexeme_transcripts']))

    #if the length of the age_list is 1, the corpus does not need to be split, so everything can be dumped into 1 single file
    if len(age_list) == 1:
        print("starting to dump split")
        tokens = []
        lexemes = []
        for a, t, l in zipped:
            if isinstance(a[0], int):
                token_transcripts = [t]
                lexeme_transcripts = [l]
                for t, l in zip(token_transcripts, lexeme_transcripts):
                    tokens.extend(t)  # this is now a list of utterances
                    lexemes.extend(l) # this is now a list of corresponding lexemes to each utterance
        save = file_dir + '/ChildDirected.json'
        json.dump((tokens, lexemes), open(save, 'w'))

    #if the length of the age_list is larger than one, then split the corpus according to age, which these loops do
    elif len(age_list) > 1:
        for i in range(len(age_list)):
            print("for child directed speech, starting to dump split: ", str(age_list[i]))
            if i == 0:
                tokens = []
                lexemes = []
                for a, t, l in zipped:
                    if isinstance(a[0], int):
                        if a[0] < int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokens.extend(t)  # this is now a list of utterances
                                lexemes.extend(l) # this is now a list of corresponding lexemes to each utterance
                save = file_dir + '/ChildDirected' + str(age_list[i]) + '.json'
                json.dump((tokens, lexemes), open(save, 'w'))
            elif i > 0 and i < (len(age_list)-1):
                tokens = []
                lexemes = []
                for a, t, l in zipped:
                    if isinstance(a[0], int):
                        if a[0] >= int(age_list[i-1]) and a[0] < int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokens.extend(t)  # this is now a list of utterances
                                lexemes.extend(l) # this is now a list of corresponding lexemes to each utterance
                save = file_dir + '/ChildDirected' + str(age_list[i]) + '.json'
                json.dump((tokens, lexemes), open(save, 'w'))
            elif i == (len(age_list)-1):
                tokensMinus = []
                lexemesMinus = []
                tokensPlus = []
                lexemesPlus = []
                for a, t, l in zipped:
                    if isinstance(a[0], int):
                        if a[0] >= int(age_list[i-1]) and a[0] < int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokensMinus.extend(t)  # this is now a list of utterances
                                lexemesMinus.extend(l) # this is now a list of corresponding lexemes to each utterance
                        if a[0] >= int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokensPlus.extend(t)  # this is now a list of utterances
                                lexemesPlus.extend(l) # this is now a list of corresponding lexemes to each utterance
                saveMinus = file_dir + '/ChildDirected' + str(age_list[i]) + '.json'
                json.dump((tokensMinus, lexemesMinus), open(saveMinus, 'w'))
                savePlus = file_dir + '/ChildDirected' + str(age_list[i]) + 'Plus.json'
                json.dump((tokensPlus, lexemesPlus), open(savePlus, 'w'))
    
    print("done dumping corpora")

def save_corpus_child_produced(file_dir, age_list, corpus):
    
    """
    This is a function to extract 2 lists from the .json file which was generated with process_and_save_corpus_data()
    The first list is a list of all tokens produced by target children, the second list their corresponding lexemes
    These lists are dumped in .json files, according to the age of the target child
    """
    
    # opens the previous .json file and zips the necessary parts of it in zipped
    corpus_dir = file_dir + '/%s_raw.json' % corpus
    cor = json.load(open(corpus_dir, 'r'))
    zipped = list(zip(cor['child_ages'], cor['child_token_transcripts'], cor['child_lexeme_transcripts']))
    
    #if the length of the age_list is 1, the corpus does not need to be split, so everything can be dumped into 1 single file
    if len(age_list) == 1:
        print("starting to dump corpus")
        tokens = []
        lexemes = []
        for a, t, l in zipped:
            if isinstance(a[0], int):
                token_transcripts = [t]
                lexeme_transcripts = [l]
                for t, l in zip(token_transcripts, lexeme_transcripts):
                    tokens.extend(t)  # this is now a list of utterances
                    lexemes.extend(l) # this is now a list of corresponding lexemes to each utterance
        save = file_dir + '/ChildProduced.json'
        json.dump((tokens, lexemes), open(save, 'w'))

    #if the length of the age_list is larger than one, then split the corpus according to age, which these loops do
    elif len(age_list) > 1:
        for i in range(len(age_list)):
            print("for child produced speech, starting to dump split: ", str(age_list[i]))
            if i == 0:
                tokens = []
                lexemes = []
                for a, t, l in zipped:
                    if isinstance(a[0], int):
                        if a[0] < int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokens.extend(t)  # this is now a list of utterances
                                lexemes.extend(l) # this is now a list of corresponding lexemes to each utterance
                save = file_dir + '/ChildProduced' + str(age_list[i]) + '.json'
                json.dump((tokens, lexemes), open(save, 'w'))
            elif i > 0 and i < (len(age_list)-1):
                tokens = []
                lexemes = []
                for a, t, l in zipped:
                    if isinstance(a[0], int):
                        if a[0] >= int(age_list[i-1]) and a[0] < int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokens.extend(t)  # this is now a list of utterances
                                lexemes.extend(l) # this is now a list of corresponding lexemes to each utterance
                save = file_dir + '/ChildProduced' + str(age_list[i]) + '.json'
                json.dump((tokens, lexemes), open(save, 'w'))
            elif i == (len(age_list)-1):
                tokensMinus = []
                lexemesMinus = []
                tokensPlus = []
                lexemesPlus = []
                for a, t, l in zipped:
                    if isinstance(a[0], int):
                        if a[0] >= int(age_list[i-1]) and a[0] < int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokensMinus.extend(t)  # this is now a list of utterances
                                lexemesMinus.extend(l) # this is now a list of corresponding lexemes to each utterance
                        if a[0] >= int(age_list[i]):
                            token_transcripts = [t]
                            lexeme_transcripts = [l]
                            for t, l in zip(token_transcripts, lexeme_transcripts):
                                tokensPlus.extend(t)  # this is now a list of utterances
                                lexemesPlus.extend(l) # this is now a list of corresponding lexemes to each utterance
                saveMinus = file_dir + '/ChildProduced' + str(age_list[i]) + '.json'
                json.dump((tokensMinus, lexemesMinus), open(saveMinus, 'w'))
                savePlus = file_dir + '/ChildProduced' + str(age_list[i]) + 'Plus.json'
                json.dump((tokensPlus, lexemesPlus), open(savePlus, 'w'))
    
    print("done dumping corpora")

if __name__ == '__main__':
    file_dir = input("Please provide the path to the folder in which all newly created files will be saved \n") #C:/Users/wgvan/Documents/Universiteit_Nijmegen/jaar2/thesis/CHILDES_Corpus/Test

    if not os.path.exists(file_dir):
        raise ValueError("The path to the folder does not exist, please provide a valid path!")

    root_folder = input("Please provide the root folder in which all .xml files for the CHILDES corpus are stored. \n") #C:/Users/wgvan/Documents/Universiteit_Nijmegen/jaar2/thesis/CHILDES_Corpus/Childes_Corpus

    if not os.path.exists(root_folder):
        raise ValueError("The path to the root folder does not exist, please provide a valid path!")

    print("You can now specify at what ages you want to split the corpus, if you do not want to split the corpus, just provide the number 1000.")
    age_separator = input("Please specify a list in ascending order of whole numbers (looks like: 24, 36, etc.), specifying at which childs age (in months), the corpus should be split. \n") #24, 30, 36, 42, 48, 54, 60, 72
    age_list = list(map(int, age_separator.split(','))) #[24, 30, 36, 42, 48, 54, 60, 72]
    if len(age_list) == 0:
        raise ValueError("You have to at least specify one age! If you don't want to split the corpus, just provide the number 1000.")
    for i in age_list:
        if not isinstance(i, int):
            raise ValueError("Please provide whole numbers only!")
    if len(age_list) > 1:
        for i in range(len(age_list)-1):
            if age_list[i] >= age_list[i+1]:
                raise ValueError("Please provide the numbers in a sequential order and do not repeat the same number!")
        

    filelist = []    # list of files of all corpora
    filelist2 = []    # trimmed filelist of all corpora

    # use the glob function to to look through all depths of the childes corpus files
    # before using this, check your map structure and at which depths the .xml files exist relative to the root corpus (in this case EnglishUS)
    for file in glob.glob(root_folder + "/*.xml"):
        filelist.append(file)
    for file in glob.glob(root_folder + "/*/*.xml"):
        filelist.append(file)
    for file in glob.glob(root_folder + "/*/*/*.xml"):
        filelist.append(file)
    for file in glob.glob(root_folder + "/*/*/*/*.xml"):
        filelist.append(file)
    for file in glob.glob(root_folder + "/*/*/*/*/*.xml"):
        filelist.append(file)
    for file in glob.glob(root_folder + "/*/*/*/*/*/*.xml"):
        filelist.append(file)

    # this loop trims the root folder name
    # this is done to be able to use the root folder name in the code below
    for file in filelist:
        file2 = file.replace(str(root_folder) + '\\', '')
        file3 = file2.replace('\\', '/')
        filelist2.append(file3)

    #process_and_save_corpus_data(filelist2, file_dir, root_folder)
    save_corpus_child_directed(file_dir, age_list, corpus='corpus_EN_child')
    save_corpus_child_produced(file_dir, age_list, corpus='corpus_EN_child')