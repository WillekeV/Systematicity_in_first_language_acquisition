from collections import defaultdict
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity

def _mp_compute_cossim(args):

    return compute_cossim(*args)

def compute_cossim(target_space, original_space, word):

    cossim = float(cosine_similarity(target_space[word], original_space.get_vector(word)))
    return word, cossim

def compute_cosine_similarity(target_space, original_space, target_vocab, threads = 8):

    """
    :param target_space:    dict containing the wordvectors of the target vocab computed using LDL
    :param original_space:  the original SemanticSpace object of the target vocab
    :param target_vocab:    list, target words for which to compute snd.
    :param threads:         int, the number of cores to use for parallel processing.
    :return:                dict, mapping words to their respective cosine similarity values. Higher values indicate more similarity
    """

    word_to_cossim = defaultdict(float)
    target_vocab = target_vocab

    with mp.Pool(threads) as pool:
        outputs = pool.imap(_mp_compute_cossim, ((target_space, original_space, word) for word in target_vocab))
        for word, cossim in outputs:
            word_to_cossim[word] = cossim

    return word_to_cossim


