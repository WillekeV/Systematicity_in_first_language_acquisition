import heapq
import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
import pandas as pd
from scipy import spatial



def _mp_compute_snd(args):

    return compute_snd(*args)


def compute_snd(embeddings, reference_wordvecs, n_neighbors, w):

    wordvec_w = embeddings.word_vector(w)

    nlargest = heapq.nlargest(
        n_neighbors,
        [1 - spatial.distance.cosine(wordvec_w, wv) for wv in reference_wordvecs])

    snd = np.mean(nlargest)

    return w, snd


def neighborhood_density(embeddings, reference_space, target_words, reference_words, n_neighbors=20, threads=8):

    """
    :param embeddings:      a SemanticSpace object of the target vocab
    :param reference_space: a SemeanticSpace object of the reference vocab
    :param target_words:    list, target words for which to compute snd.
    :param reference_words: list, reference vocabulary listing words to consider as valid neighbors.
    :param n:               int, number of neighbours to consider. Default to 20.
    :param threads:         int, the number of cores to use for parallel processing.
    :return:                dict, mapping words to their respective SND values. Higher values indicate sparser semantic
                            neighborhoods
    """

    w2snd = defaultdict(float)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started computing semantic neighborhood density..."))

    begintime = time.time()

    reference_wordvecs = []
    for word in reference_words:
        reference_wordvecs.append(reference_space.word_vector(word))

    with mp.Pool(threads) as pool:
        outputs = pool.imap(_mp_compute_snd, ((embeddings, reference_wordvecs, n_neighbors, word) for word in target_words))
        for w, snd in outputs:
            w2snd[w] = snd

    endtime = time.time()

    print("elapsed: ", endtime - begintime)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done."))

    return w2snd
