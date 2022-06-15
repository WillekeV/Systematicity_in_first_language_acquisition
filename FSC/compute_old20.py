from old20 import old_n

def calculate_old20(target_words, reference_words):

    """
    :param target_words:    list, target words for which to compute old20.
    :param reference_words: list, reference vocabulary listing words to consider as valid neighbors.
    :param threads:         int, the number of cores to use for parallel processing.
    :return:                dict, mapping words to their respective old20 values.
    """

    target_words = list(target_words)

    value_old20 = old_n(target_words, reference_words, n=20)

    value_old20 = list(value_old20)

    w2old20 = dict(zip(target_words, value_old20))

    return w2old20