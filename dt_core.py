# version 1.0
import math
from typing import List
from anytree import Node, RenderTree

import dt_global
import dt_provided

import numpy as np  # numpy==1.19.2

def find_all_labels(examples, pos, value, feature, label_set, direction):

    label_set.add(examples[pos][dt_global.label_index])

    next_pos = pos + direction
    while (next_pos < len(examples) and next_pos >= 0 and examples[next_pos][feature] == value):
        label_set.add(examples[next_pos][dt_global.label_index])
        next_pos += direction
    return

def diff_sets(set1, set2):
    for a in set1:
        if a not in set2:
            return True
    for a in set2:
        if a not in set1:
            return True
    return False


def get_splits(examples: List, feature: str): #-> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.
    
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """ 
    feature_index = dt_global.feature_names.index(feature)
    possible_split_points = []
    forward = 1
    backward = -1
    
    if (len(examples) < 2): 
        return possible_split_points

    examples.sort(key=lambda x: x[feature_index])
    for i in range(len(examples) - 1):
        curr_value = examples[i][feature_index]
        next_value = examples[i + 1][feature_index]
        if (curr_value != next_value):
            curr_set = set()
            next_set = set()
            find_all_labels(examples, i, curr_value, feature_index, curr_set, backward)
            find_all_labels(examples, i + 1, next_value, feature_index, next_set, forward)
            if diff_sets(curr_set, next_set):
                possible_split_points.append((curr_value + next_value) / 2)
    return possible_split_points


def simple_entropy(d, lenth):
    ret = 0.0
    for i in d.keys():
        proportion = d[i] / lenth
        if (proportion != 0):
            ret += proportion * math.log2(proportion)
    return -ret


def calculate_entropy(start, end, examples, class_dict):
    
    for i in range(start, end):
        class_dict[examples[i][dt_global.label_index]] += 1
    lenth = end - start
    ret = 0.0
    for i in class_dict.keys():
        proportion = class_dict[i] / lenth
        if (proportion != 0.0):
            ret += proportion * math.log2(proportion)
    return -ret

def choose_best_split(examples:List, feature: str, ret:List, class_dict) -> (str, float, float):
    # print(feature)
    feature_index = dt_global.feature_names.index(feature)
    lenth = len(examples)
    list_of_splits = get_splits(examples, feature)
    # print(examples)
    los = len(list_of_splits)
    entropy_before = calculate_entropy(0, lenth, examples, class_dict)
    # print("class_dict: ", class_dict, "lenth: ", lenth)
    
    current_split = 0

    # count the numbers here good

    left_dict = {}
    for k in class_dict.keys():
        left_dict[k] = 0

    for i in range(lenth - 1):
        if current_split == los:
            break

        left_dict[examples[i][dt_global.label_index]] += 1

        if (dt_provided.less_than_or_equal_to(examples[i][feature_index], list_of_splits[current_split]) and \
            dt_provided.less_than_or_equal_to(list_of_splits[current_split], examples[i + 1][feature_index])):

            # print("i: ", i, "lenth:", lenth)
            # print("left_dict: ", left_dict, "lenth: ", i + 1)
            left_entropy = simple_entropy(left_dict, i + 1)

            right_dict = {}
            for k in class_dict.keys():
                right_dict[k] = class_dict[k] - left_dict[k]

            # print("i: ", i, "lenth:", lenth)
            # print("right_dict: ", right_dict, "lenth: ", lenth - i - 1)
            entropy_after = (i + 1) / lenth * left_entropy + \
                (lenth - i - 1) / lenth * simple_entropy(right_dict, lenth - i - 1)



            info_gain = entropy_before - entropy_after
            if (dt_provided.less_than(ret[2], info_gain)):
                ret[0] = feature
                ret[1] = list_of_splits[current_split]
                ret[2] = info_gain
            current_split += 1


def choose_feature_split(examples: List, features: List[str]) -> (str, float, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None, -1, and -inf.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature, the best split value, the max expected information gain
    :rtype: str, float, float
    """ 

    ret_vals = [None, -1, -math.inf]

    class_set = set(np.array(examples)[:, -1])
    class_dict = {}

    for i in range(len(features)):
        for c in class_set:
            class_dict[c] = 0.0
        choose_best_split(examples, features[i], ret_vals, class_dict)

    return ret_vals[0], ret_vals[1], ret_vals[2]


def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """ 

    feature_index = dt_global.feature_names.index(feature)

    first_list = []
    second_list = []
    for i in range(len(examples)):
        if dt_provided.less_than_or_equal_to(examples[i][feature_index], split):
            first_list.append(examples[i])
        else:
            second_list.append(examples[i])
    return first_list, second_list

def dict_init(examples, dict):
    for i in range(len(examples)):
        if examples[i][dt_global.label_index] in dict.keys():
            dict[examples[i][dt_global.label_index]] += 1
        else:
            dict[examples[i][dt_global.label_index]] = 1


def select_majority(dict):
    major_class = -1
    major_class_count = 0
    for v in dict.keys():
        if dt_provided.less_than(major_class_count, dict[v]) or \
        (dt_provided.less_than_or_equal_to(major_class_count, dict[v]) and \
        dt_provided.less_than(v, major_class)):
            major_class = v
            major_class_count = dict[v]
    return major_class

def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """ 

    dict = {}
    dict_init(examples, dict)

    cur_node.majority = select_majority(dict)

    if (cur_node.depth >= max_depth):
        cur_node.decision = cur_node.majority
        return

    best_feature, best_split_value, max_info_gain = choose_feature_split(examples, features)

    if (best_feature == None):
        cur_node.decision = cur_node.majority
        return


    cur_node.feature = best_feature
    cur_node.split = best_split_value
    cur_node.max_info_gain = max_info_gain
    left_child = Node(name=cur_node.name + "0", parent=cur_node, depth=cur_node.depth + 1)
    right_child = Node(name=cur_node.name + "1", parent=cur_node, depth=cur_node.depth + 1)

    # print(RenderTree(cur_node))    

    left_examples, right_examples = split_examples(examples, best_feature, best_split_value)
    split_node(left_child, left_examples, features, max_depth)
    split_node(right_child, right_examples, features, max_depth)

def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """ 
    root = Node(name="root", depth=0)
    split_node(root, examples, features, max_depth)
    # print(RenderTree(root))
    return root

def example_is_less(cur_node, example):
    test_val = example[dt_global.feature_names.index(cur_node.feature)]
    return dt_provided.less_than_or_equal_to(test_val, cur_node.split)


def predict(cur_node: Node, example, max_depth=math.inf) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the decision for the given example
    :rtype: int
    """ 
    
    if cur_node.is_leaf:
        return cur_node.decision
    elif max_depth == cur_node.depth:
        return cur_node.majority    
    elif example_is_less(cur_node, example):
        return predict(cur_node.children[0], example, max_depth)
    return  predict(cur_node.children[1], example, max_depth)



def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth,
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """ 
    noe = len(examples)
    if noe == 0:
        return 0.0;
    nor = 0;
    for i in range(noe):
        if predict(cur_node, examples[i], max_depth) == examples[i][dt_global.label_index]:
            nor += 1
    return nor / noe


def post_prune_helper(cur_node, min_info_gain, loc):
    if cur_node.is_leaf:
        return
    if cur_node.children[0].is_leaf and cur_node.children[1].is_leaf and \
    dt_provided.less_than(cur_node.max_info_gain, min_info_gain):
        cur_node.decision = cur_node.majority
        cur_node.children = []
        post_prune_helper(cur_node.parent, min_info_gain, loc)
        loc[0] = False
        return
    else:
        post_prune_helper(cur_node.children[0], min_info_gain, loc)
        post_prune_helper(cur_node.children[1], min_info_gain, loc)


def post_prune(cur_node: Node, min_info_gain: float):
    """
    Given a tree with cur_node as the root, and the minimum information gain,
    post prunes the tree using the minimum information gain criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the information gain at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the information gain at every leaf parent is greater than
    or equal to the pre-defined value of the minimum information gain.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_info_gain: the minimum information gain
    :type min_info_gain: float
    """
    if cur_node.is_leaf:
        return
    if cur_node.children[0].is_leaf and cur_node.children[1].is_leaf and \
    dt_provided.less_than(cur_node.max_info_gain, min_info_gain):
        cur_node.decision = cur_node.majority
        cur_node.children = []
        if cur_node.is_root == False:
            post_prune(cur_node.parent, min_info_gain)
        return
    else:
        post_prune(cur_node.children[0], min_info_gain)
        if len(cur_node.children) != 0:
            post_prune(cur_node.children[1], min_info_gain)


def pre_prune(cur_node, depth):
    if cur_node.is_leaf:
        return
    elif cur_node.depth >= depth:
        cur_node.decision = cur_node.majority
        cur_node.children = []
        return
    else:
        pre_prune(cur_node.children[0], depth)
        pre_prune(cur_node.children[1], depth)
    return
