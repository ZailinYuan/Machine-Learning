# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time


# Self - defined functions:

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def stat(vec):
    """
    This function group vector values by unique values it has, return a dictionary in a
    form of (unique_value : counts). This function is in convince of calculating probabilities.

    :param vec: any vector
    :return: dictionary that statistic occurrence of each unique values.
    """
    unique = {}
    for i in vec:
        if i not in unique:
            unique[i] = 1
        else:
            new_val = unique.get(i)
            new_val = new_val + 1
            unique[i] = new_val
    return unique


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    unique = {}

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    # unique: value - frequency
    # hz: entropy of the vector
    unique = stat(y)
    hz = 0
    for i in unique:
        p = unique[i]/len(y)
        hz = hz + (p * np.log2(p))
    return -hz


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    px = stat(x)
    py = stat(y)

    # Joint probability:
    joint_stat = {}
    for X in x:
        for Y in y:
            key = (X, Y)
            if key not in joint_stat:
                joint_stat[key] = 1
            else:
                new_val = joint_stat[key] + 1
                joint_stat[key] = new_val

    I_XY = 0
    size = len(x) * len(y)
    for i in px:
        for j in py:
            p_x = px[i] / len(x)
            p_y = py[j] / len(y)
            p_xy = joint_stat[(i, j)] / size
            I_XY = p_xy * np.log2(p_xy / p_x / p_y)
    return I_XY


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}
        },
     (4, 1, True): 1}
    """

    #Terminate condition 3:
    if y.sum == len(y):
        return 1
    if y.sum == 0:
        return 0

    # Terminate condition 1:
    if depth == max_depth:
        if y.sum() < (len(y)/2):
            return 0
        elif y.sum() == (len(y)/2):
            # Randomly pick up a value:
            return y[0]
        else:
            return 1


    # Current node entropy:
    cur_entropy = entropy(y)

    # If no attribute_value_pairs, generate one:
    pairs = []
    if attribute_value_pairs is None:
        # Candidate nodes:
        rows = x.shape[0]
        cols = x.shape[1]
        for col in range(cols):
            vals = set()
            for i in x[:,col]:
                vals.add(i)
            for val in vals:
                pairs.append((col, val))
    else:
        pairs = attribute_value_pairs

    # Terminate condition 2:
    if len(pairs) == 0:
        if y.sum() < (len(y) / 2):
            return 0
        elif y.sum() == (len(y) / 2):
            # Randomly pick up a value:
            return y[0]
        else:
            return 1

    # If a attribute_value_pairs is here, select from it the best attribute:
    info_gains = {}
    max_info_gain = 0
    for attribute_pair in pairs:
        attribute = attribute_pair[0]

        # Calculate population of separation of the attribute:
        positive = 0
        negative = 0
        pos = 0
        y_1 = np.array([])
        y_2 = np.array([])
        for a in x[:, attribute]:
            if a == attribute_pair[1]:
                positive = positive + 1
                y_1 = np.append(y_1, y[pos])
                pos = pos + 1
            else:
                negative = negative + 1
                y_2 = np.append(y_2, y[pos])
                pos = pos + 1
        weight_1 = positive/len(y)
        weight_2 = negative/len(y)
        entropy_1 = entropy(y_1)
        entropy_2 = entropy(y_2)

        info_gain = cur_entropy - (weight_1 * entropy_1 + weight_2 * entropy_2)
        info_gains[info_gain] = attribute_pair
        if max_info_gain < info_gain:
            max_info_gain = info_gain

    attribute_chozen = info_gains[max_info_gain]
    pairs.remove(attribute_chozen)

    # Parameter for the next recursive call: x, y, attribute_value_pairs=None, depth=0, max_depth=5
    attribute = attribute_chozen[0]
    sub_y_true = np.array([])
    sub_x_true = np.empty([0,x.shape[1]])
    sub_y_false = np.array([])
    sub_x_false = np.empty([0,x.shape[1]])
    for i in range(len(y)):
        if x[i, attribute] == attribute_chozen[1]:
            sub_x_true = np.row_stack((sub_x_true, x[i,:]))
            sub_y_true = np.append(sub_y_true, y[i])
        else:
            sub_x_false = np.row_stack((sub_x_false, x[i,:]))
            sub_y_false = np.append(sub_y_false, y[i])

    tuple1 = (attribute_chozen[0], attribute_chozen[1], True)
    tuple2 = (attribute_chozen[0], attribute_chozen[1], False)

    depth = depth + 1

    if len(sub_y_true) == 0:
        return {tuple1: random.randint(0,1), tuple2: id3(sub_x_false, sub_y_false, pairs, depth, max_depth)}
    elif len(sub_x_false) == 0:
        return {tuple1: id3(sub_x_true, sub_y_true, pairs, depth, max_depth), tuple2: random.randint(0,1)}
    else:
        return {tuple1: id3(sub_x_true, sub_y_true, pairs, depth, max_depth), tuple2: id3(sub_x_false, sub_y_false, pairs, depth, max_depth)}


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # Take out the node:
    node = {}
    for i in tree:
        node = i
        break

    if x[node[0]] == node[1]:
        sub_key = (node[0], node[1], True)
    else:
        sub_key = (node[0], node[1], False)

    sub_tree = tree[sub_key]
    if isinstance(sub_tree, int) or isinstance(sub_tree, float):
        return sub_tree
    else:
        return predict_example(x, sub_tree)


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    size = len(y_true)
    error_sum = 0
    for i in range(size):
        if y_true[i] != y_pred[i]:
            error_sum = error_sum + 1
    return error_sum/size


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':

    problems = ['monks_data/monks-1.train', 'monks_data/monks-2.train', 'monks_data/monks-3.train']
    tests = ['monks_data/monks-1.test', 'monks_data/monks-2.test', 'monks_data/monks-3.test']

    # For each problem:
    size = len(problems)
    for i in range(size):
        # Load the training data
        M = np.genfromtxt(problems[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(tests[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        # In each problem, for every depth:
        depth = np.array([])
        tst_error = np.array([])
        for j in range(1,11):
            # Learn a decision tree of depth 1 - 10
            decision_tree = id3(Xtrn, ytrn, max_depth=j)

            # Pretty print it to console
            pretty_print(decision_tree)

            if i == 1:
                # Visualize the tree and save it as a PNG image
                dot_str = to_graphviz(decision_tree)
                render_dot_file(dot_str, './my_learned_tree_depth_' + str(j))

            # Compute the test error
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)

            # Confusion matrix:
            if i == 1:
                class_names = [0,1]
                np.set_printoptions(precision=2)
                y_pred = np.array(y_pred)
                plot_confusion_matrix(ytst, y_pred, classes=class_names,
                                      title='Confusion matrix, without normalization')
                plot_confusion_matrix(ytst, y_pred, classes=class_names,
                                      title='Normalized confusion matrix')

            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
            depth = np.append(depth, j)
            tst_error = np.append(tst_error, tst_err)

        # Plot depth - test error curve:
        plt.figure()
        plt.ylim(ymin=0)
        plt.ylim(ymax=1)
        plt.plot(depth, tst_error, 12, marker='o', color='orange')
        plt.title('Monk_' + str(i+1), fontsize=18)
        plt.xlabel('Depth', fontsize=16)
        plt.ylabel('Test Error', fontsize=16)
        plt.show()