from collections import defaultdict
import multiprocessing as mp
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

target_embeddings = None
original_embeddings = None

def _mp_compute_cosdist(args):

    return compute_cosdist(*args)

def compute_cosdist(i, word):

    # cos = float(cosine(target_embeddings[word], original_embeddings.get_vector(word)))
    cos = float(cosine_similarity(target_embeddings[word], original_embeddings.get_vector(word)))

    return word, cos

def compute_cosine_distance(target_space, original_space, target_vocab, threads = 64):

    """
    :param target_space:    dict containing the wordvectors of the target vocab computed using LDL
    :param original_space:  the original SemanticSpace object of the target vocab
    :param target_vocab:    list, target words for which to compute snd.
    :param threads:         int, the number of cores to use for parallel processing.
    :return:                dict, mapping words to their respective cosine similarity values. Higher values indicate more similarity
    """

    global target_embeddings
    global original_embeddings

    target_embeddings = target_space
    original_embeddings = original_space

    word_to_cosdist = defaultdict(float)
    target_vocab = target_vocab

    with mp.Pool(threads) as pool:
        outputs = pool.imap(_mp_compute_cosdist, ((i, word) for i, word in enumerate(target_vocab)))
        for word, cossim in outputs:
            word_to_cosdist[word] = cossim

    return word_to_cosdist


