# importing pandas
import pandas as pd

# importing numpy
import numpy as np

import math

from collections import Counter


def read_csv_file(filename=''):
    complete_data = pd.read_csv(filename)
    print 'file read ...'
    return complete_data


def get_data_header(data):
    data_header = list(data.columns.values)
    return data_header


def get_class_of_data(data):
    class_of_data = data.iloc[0:, -1]
    return class_of_data


def get_max_col_value(column):
    return column.max(axis=0)


def get_min_col_value(column):
    return column.min(axis=0)


def new_value_generator(data_point, max_value, min_value):
    new_value = (data_point - min_value) / (max_value - min_value)
    return new_value


def data_normalizer(data):
    print 'Normalizing Data ...'
    for column_number in range(0, data.shape[1] - 1):
        max_value = get_max_col_value(data.iloc[0:, column_number])
        min_value = get_min_col_value(data.iloc[0:, column_number])
        data.iloc[:, column_number] = data.iloc[:, column_number].apply(new_value_generator, args=(max_value, min_value))
    return data


def k_fold_cross_validation(data, k_folds):
    print 'Creating Folds ...'
    number_of_examples = data.shape

    number_of_examples_per_partition = number_of_examples[0] / k_folds

    remainder = number_of_examples[0] % k_folds

    data['partition_no'] = np.nan

    row_lst = np.random.choice(number_of_examples[0], size=(1, number_of_examples[0]), replace=False).tolist()[0]

    if remainder == 0:
        for fold in range(0, k_folds):
            for i in range(0, number_of_examples_per_partition):
                data.set_value(row_lst[0], 'partition_no', fold)
                row_lst.pop(0)
    else:
        for fold in range(0, k_folds):
            if remainder > 0:
                compensation = 1
                remainder = remainder - 1
            else:
                compensation = 0

            for i in range(0, number_of_examples_per_partition + compensation):
                data.set_value(row_lst[0], 'partition_no', fold)
                row_lst.pop(0)
    return data


def get_folded_data(data, value):
    tt_data = data.loc[data['partition_no'] == int(value)]
    test_data = tt_data.iloc[0:, 0:-1]
    tn_data = data.loc[data['partition_no'] != int(value)]
    train_data = tn_data.iloc[0:, 0:-1]
    return train_data, test_data


def max_class_counter(lst):
    unique, pos = np.unique(lst, return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    return unique[maxpos]


def knn_accuracy_kernel_euclidean(train_data_input, test_data_input, nn_value=1):
    train_data_as_matrix = train_data_input.as_matrix()
    test_data_as_matrix = test_data_input.as_matrix()

    distance_matrix = np.zeros(shape=(test_data_as_matrix.shape[0], train_data_as_matrix.shape[0]))

    vector_row = 0
    for test_vector in test_data_as_matrix:
        vector_column = 0
        for train_vector in train_data_as_matrix:
            distance_matrix[vector_row][vector_column] = np.sqrt(np.sum(test_vector[0:(test_data_as_matrix.shape[1] - 2)] - train_vector[0:(train_data_as_matrix.shape[1] - 2)]) ** 2)
            vector_column = vector_column + 1
        vector_row = vector_row + 1

    distance_matrix_sorted_indexes = np.argsort(distance_matrix, axis=1)

    correct_prediction = 0
    if nn_value == 1:
        for tt_vector, dist_vector in zip(test_data_as_matrix, distance_matrix_sorted_indexes):
            if tt_vector[-1] == train_data_as_matrix[dist_vector[0]][-1]:
                correct_prediction = correct_prediction + 1
    else:

        slice_of_distance_matrix_sorted_indexes = distance_matrix_sorted_indexes[0:, 0:nn_value]

        for test_vector, distance_vector in zip(test_data_as_matrix, slice_of_distance_matrix_sorted_indexes):
            class_of_test_vector = test_vector[-1]
            list_of_closest_train_class = list()
            for train_vector_index in distance_vector:
                list_of_closest_train_class.append(train_data_as_matrix[train_vector_index][-1])
            # if class_of_test_vector == max(set(list_of_closest_train_class), key=list_of_closest_train_class.count):
            if class_of_test_vector == max_class_counter(list_of_closest_train_class):
                correct_prediction = correct_prediction + 1

    knn_euclidean_fold_accuracy = (correct_prediction / float(test_data_as_matrix.shape[0])) * 100
    return knn_euclidean_fold_accuracy


def knn_accuracy_kernel_rbf(train_data_input, test_data_input, nn_value=1):
    train_data_as_matrix = train_data_input.as_matrix()
    test_data_as_matrix = test_data_input.as_matrix()

    distance_matrix = np.zeros(shape=(test_data_as_matrix.shape[0], train_data_as_matrix.shape[0]))

    vector_row = 0
    for test_vector in test_data_as_matrix:
        vector_column = 0
        for train_vector in train_data_as_matrix:
            euclidean_distance = np.sqrt(np.sum(test_vector[0:(test_data_as_matrix.shape[1] - 2)] - train_vector[0:(train_data_as_matrix.shape[1] - 2)]) ** 2)
            squared_euclidean_distance = euclidean_distance ** 2
            sigma = 0.5
            sigma_squared = sigma ** 2
            #kernel_calculated_distance = np.exp(-0.5 * (squared_euclidean_distance / sigma_squared))
            #kernel_calculated_distance = math.exp(-0.5 * (squared_euclidean_distance / sigma_squared))
            kernel_calculated_distance = 2.0*(1.0 - math.exp(-1 * (squared_euclidean_distance / sigma_squared)))
            # kernel_calculated_distance = math.exp(-squared_euclidean_distance)
            distance_matrix[vector_row][vector_column] = kernel_calculated_distance
            vector_column = vector_column + 1
        vector_row = vector_row + 1

    distance_matrix_sorted_indexes = np.argsort(distance_matrix, axis=1)

    correct_prediction = 0
    if nn_value == 1:
        for tt_vector, dist_vector in zip(test_data_as_matrix, distance_matrix_sorted_indexes):
            if tt_vector[-1] == train_data_as_matrix[dist_vector[0]][-1]:
                correct_prediction = correct_prediction + 1
    else:

        slice_of_distance_matrix_sorted_indexes = distance_matrix_sorted_indexes[0:, 0:nn_value]

        for test_vector, distance_vector in zip(test_data_as_matrix, slice_of_distance_matrix_sorted_indexes):
            class_of_test_vector = test_vector[-1]
            list_of_closest_train_class = list()
            for train_vector_index in distance_vector:
                list_of_closest_train_class.append(train_data_as_matrix[train_vector_index][-1])
            # if class_of_test_vector == max(set(list_of_closest_train_class), key=list_of_closest_train_class.count):
            if class_of_test_vector == max_class_counter(list_of_closest_train_class):
                correct_prediction = correct_prediction + 1

    knn_rbf_fold_accuracy = (correct_prediction / float(test_data_as_matrix.shape[0])) * 100
    return knn_rbf_fold_accuracy


def knn_accuracy_kernel_polynomial(train_data_input, test_data_input, nn_value=1):
    train_data_as_matrix = train_data_input.as_matrix()
    test_data_as_matrix = test_data_input.as_matrix()

    distance_matrix = np.zeros(shape=(test_data_as_matrix.shape[0], train_data_as_matrix.shape[0]))

    vector_row = 0
    for test_vector in test_data_as_matrix:
        vector_column = 0
        for train_vector in train_data_as_matrix:
            p = 6
            kxx = (1 + np.dot(test_vector[0:(test_data_as_matrix.shape[1] - 2)], test_vector[0:(test_data_as_matrix.shape[1] - 2)])) ** p
            kyy = (1 + np.dot(train_vector[0:(train_data_as_matrix.shape[1] - 2)], train_vector[0:(train_data_as_matrix.shape[1] - 2)])) ** p
            kxy = (1 + np.dot(test_vector[0:(test_data_as_matrix.shape[1] - 2)], train_vector[0:(train_data_as_matrix.shape[1] - 2)])) ** p
            kernel_calculated_distance = kxx - 2*kxy + kyy
            distance_matrix[vector_row][vector_column] = kernel_calculated_distance
            vector_column = vector_column + 1
        vector_row = vector_row + 1

    #distance_matrix_sorted_indexes = np.fliplr(np.argsort(distance_matrix, axis=1))
    distance_matrix_sorted_indexes = np.argsort(distance_matrix, axis=1)

    correct_prediction = 0
    if nn_value == 1:
        for tt_vector, dist_vector in zip(test_data_as_matrix, distance_matrix_sorted_indexes):
            if tt_vector[-1] == train_data_as_matrix[dist_vector[0]][-1]:
                correct_prediction = correct_prediction + 1
    else:

        slice_of_distance_matrix_sorted_indexes = distance_matrix_sorted_indexes[0:, 0:nn_value]

        for test_vector, distance_vector in zip(test_data_as_matrix, slice_of_distance_matrix_sorted_indexes):
            class_of_test_vector = test_vector[-1]
            list_of_closest_train_class = list()
            for train_vector_index in distance_vector:
                list_of_closest_train_class.append(train_data_as_matrix[train_vector_index][-1])
            # if class_of_test_vector == max(set(list_of_closest_train_class), key=list_of_closest_train_class.count):
            if class_of_test_vector == max_class_counter(list_of_closest_train_class):
                correct_prediction = correct_prediction + 1

    knn_polynomial_fold_accuracy = (correct_prediction / float(test_data_as_matrix.shape[0])) * 100
    return knn_polynomial_fold_accuracy


def knn_accuracy(train_data_input, test_data_input, kernel, nn_value):
    if kernel == 'euclidean':
        fold_accuracy = knn_accuracy_kernel_euclidean(train_data_input, test_data_input, nn_value)
        return fold_accuracy

    if kernel == 'polynomial':
        fold_accuracy = knn_accuracy_kernel_polynomial(train_data_input, test_data_input, nn_value)
        return fold_accuracy

    if kernel == 'rbf':
        fold_accuracy = knn_accuracy_kernel_rbf(train_data_input, test_data_input, nn_value)
        return fold_accuracy
