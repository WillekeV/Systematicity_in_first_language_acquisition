import numpy as np

def cross_mapping(train_form_matrix, train_semantic_space, test_form_matrix, target_vocab):

    """
    :param train_form_matrix:       NumPy 2d array
    :param train_semantic_space:    NumPy 2d array
    :param test_form_matrix:        NumPy 2d array
    :return:                        NumPy 2d array
    """

    subset_transform = np.dot(np.linalg.pinv(train_form_matrix), train_semantic_space)
    #subset_transform = np.nan_to_num(subset_transform)
    estimated_semantic_space = np.dot(test_form_matrix, subset_transform)
    
    #print(estimated_semantic_space)
    
    space_dict = dict()
    
    for i in range(len(estimated_semantic_space)):
        for word in target_vocab:
            space_dict[word] = estimated_semantic_space[i].reshape(1,-1)

    return space_dict