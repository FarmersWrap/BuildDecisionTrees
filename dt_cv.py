# version 1.0
from typing import List

import dt_global
import dt_core


def get_training_set(i, f):
    loe = []
    for j in range(len(f)):
        if j != i:
            loe.extend(f[j])
    return loe

def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """  

    nof = len(folds)
    novl = len(value_list)
    training_accuracy = [0.0] * novl
    validation_accuracy = [0.0] * novl
    
    value_list = value_list[::-1]
    # print(value_list)
    for i in range(nof):
        validation_set = folds[i]
        training_set = get_training_set(i, folds)
        # print("tree generation begin")
        full_tree = dt_core.learn_dt(training_set, dt_global.feature_names[:-1], value_list[0])
        # print("tree generation end")
        height = 0
        is_first = True
        first_t_acc = 0.0
        first_v_acc = 0.0 
        for j in range(novl):
            if is_first:
                dt_core.pre_prune(full_tree, value_list[j])
                height = full_tree.height
                first_t_acc = dt_core.get_prediction_accuracy(full_tree, training_set)
                first_v_acc = dt_core.get_prediction_accuracy(full_tree, validation_set)
                training_accuracy[j] += first_t_acc
                validation_accuracy[j] += first_v_acc
                is_first = False
                # print("111")
            elif value_list[j] >= height:
                training_accuracy[j] += first_t_acc
                validation_accuracy[j] += first_v_acc
                # print("222")
            else:
                # print("height: ", height, "value: ", value_list[j])
                dt_core.pre_prune(full_tree, value_list[j])
                height = value_list[j]
                training_accuracy[j] += dt_core.get_prediction_accuracy(full_tree, training_set, height)
                validation_accuracy[j] += dt_core.get_prediction_accuracy(full_tree, validation_set, height)
                # print("333")
            # print(j, training_accuracy[j], validation_accuracy[j])

    for j in range(novl):
        training_accuracy[j] = training_accuracy[j] / nof
        validation_accuracy[j] = validation_accuracy[j] / nof
    training_accuracy = training_accuracy[::-1]
    validation_accuracy = validation_accuracy[::-1]
    return training_accuracy, validation_accuracy



def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 

    nof = len(folds)
    novl = len(value_list)
    training_accuracy = [0.0] * novl
    validation_accuracy = [0.0] * novl
    
    # print(value_list)
    for i in range(nof):
        validation_set = folds[i]
        training_set = get_training_set(i, folds)
        # print("tree generation begin")
        full_tree = dt_core.learn_dt(training_set, dt_global.feature_names[:-1])
        # print("tree generation rnd")
        for j in range(novl):
                # print(value_list[j],"postprune begin")
                dt_core.post_prune(full_tree, value_list[j])
                # print("postprune end")
                training_accuracy[j] += dt_core.get_prediction_accuracy(full_tree, training_set)
                validation_accuracy[j] += dt_core.get_prediction_accuracy(full_tree, validation_set)
            # print(j, training_accuracy[j], validation_accuracy[j])
    for j in range(novl):
        training_accuracy[j] = training_accuracy[j] / nof
        validation_accuracy[j] = validation_accuracy[j] / nof
    return training_accuracy, validation_accuracy
