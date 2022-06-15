import heapq
import random
import multiprocessing as mp
from collections import defaultdict
import time
from jellyfish import levenshtein_distance
from celex import get_celex_coverage

def get_target_embedded_neighbors(targets, reference_words, celex=None):

    """
    :param targets:         list, target words OR dict, target words mapped to phonological represenation
    :param reference_words: list, reference words
    :param celex:           dict, created using the code in the celex module in this repo and used to map orthographic
                            forms to phonological forms. Default to None means neighbors are found for orthographic
                            forms
    :return:                dictionary, target words mapped to neighbours (target-embedded words)
    """

    target2neighbors = defaultdict(list)

    if celex:
        # create phonological representation of reference words
        ref_phon = {k: v for k, v in get_celex_coverage(reference_words, celex)[0]}
        # find neighbors which embed the target considering phonological forms
        for target, phon_t in targets.items():
            for ref, phon_r in ref_phon.items():
                if phon_t in phon_r and phon_t != phon_r and phon_r not in target2neighbors[target]:
                    target2neighbors[target].append(ref)
            if not target2neighbors[target]:
                # if no word embeds the target, sample 20 words to act as neighbors
                target2neighbors[target] = random.sample(reference_words, 20)
    else:
        # find neighbors which embed the target considering orthographic forms
        for word in targets:
            for ref in reference_words:
                if word in ref and word != ref and ref not in target2neighbors[word]:
                    target2neighbors[word].append(ref)
            if not target2neighbors[word]:
                # if no word embeds the target, sample 20 words to act as neighbors
                target2neighbors[word] = random.sample(reference_words, 20)

    return target2neighbors


def _mp_nearest_neighbors(args):

    return nearest_neighbors(*args)


def nearest_neighbors(n_neighbors, target, w2dist):
    nearest = []

    # early out if no items
    if len(w2dist) == 0:
        return target, []

    # fetch the distances for all neighbors from smallest to largest, then fetch the distance of the k-th neighbor
    nsmallest = heapq.nsmallest(
        n_neighbors + 1,
        [dist for _, dist in w2dist.items()])

    d = nsmallest[-1]

    # if a word has a distance lower or equal to that of the k-th neighbor, append it to the list of nearest neighbors
    for word, dis in w2dist.items():
        if 0 < dis <= d:
            nearest.append((word, dis))
    
    return target, nearest

def get_levenshtein_neighbours(targets, reference_words, celex=None, k=20, threads=8):

    """
    :param targets:         list, target words OR dict, target words mapped to phonological representation
    :param reference_words: list, reference words
    :param celex:           dict, created using the code in the celex module in this repo and used to map orthographic
                            forms to phonological forms. Default to None means neighbors are found for orthographic
                            forms
    :param k:               int, number of neighbours to consider
    :param threads:         int, indicating how many cores to spread the process over
    :return:                dict, target words mapped to k nearest neighbors according to levenshtein distance
    """

    begintime = time.time()

    target2levenshtein = defaultdict(dict)
    target2neighbors = defaultdict(list)

    if celex:
        # create phonological representation of reference words
        ref_phon = {k: v for k, v in get_celex_coverage(reference_words, celex)[0]}
        # then find neighbors according to phonological transcriptions
        for target_ortho, target_phon in targets.items():
            for reference_ortho, reference_phon in ref_phon.items():
                target2levenshtein[target_ortho][reference_ortho] = levenshtein_distance(target_phon, reference_phon)
    else:
        # find neighbours for all words in reference_words based on orthographic encoding
        for target_ortho in targets:
            for reference_ortho in reference_words:
                target2levenshtein[target_ortho][reference_ortho] = levenshtein_distance(target_ortho, reference_ortho)

    with mp.Pool(threads) as pool:
        outputs = pool.imap(
            _mp_nearest_neighbors, ((k, target, w2dist) for target, w2dist in target2levenshtein.items())
        )
        for output in outputs:
            t, nearest = output
            target2neighbors[t] = nearest

    endtime = time.time()
    print("elapsed: ", endtime - begintime)
    
    return target2neighbors
