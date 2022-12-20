from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
from decision_tree import get_maximum_information_gain_for_categorical_features


def calculate_mean(arr):
    x = 0
    for a in arr:
        x = x + a
    return x / len(arr)


def calculate_variance(arr):
    mean = calculate_mean(arr)
    total = 0
    for a in arr:
        total = total + pow((a - mean), 2)
    return total / (len(arr) - 1)


def calculate_absolut(x):
    if x >= 0:
        return x
    else:
        return x * (-1)


def feature_selection(arr):
    feature_options_list = []  # class ın her bir feature için mean, varyans ve feature için eleman sayısı
    for elem in arr:
        feature_options = []
        for feature_list in elem[1]:
            feature_options.append((round(calculate_mean(feature_list), 4),
                                    round(calculate_variance(feature_list), 4), len(feature_list)))
        feature_options_list.append((elem[0], feature_options))
    feature_count = len(arr[0][1])
    class_count = len(arr)
    se_results = []
    test_results = []
    for i in range(0, feature_count):
        se = 0
        absolut = calculate_absolut(feature_options_list[0][1][i][0] - feature_options_list[1][1][i][0])
        for j in range(0, class_count):
            se = se + (feature_options_list[j][1][i][1] / feature_options_list[j][1][i][2])  # varyans / count
        se = round(pow(se, 0.5), 4)
        se_results.append(se)
        test_results.append(round(absolut / se, 4))
    return test_results


def process(model_, train_file_name, test_file_name, optimize_binary=False, optimize_numerical=False,
            use_boxplot=False):
    df = pd.read_csv(train_file_name)
    df_test = pd.read_csv(test_file_name)
    columns = df.columns.values
    class_column = columns[0]
    all_feature_columns = columns[1: len(columns)]
    class_array = df[class_column].values
    test_class_array = df_test[class_column].values
    selected_column_indexes = []
    if optimize_numerical:
        feature_columns = []
        feature_selection_ = [['0', []], ['1', []]]
        for c in all_feature_columns:
            f = df[c].values
            zeros = []
            ones = []
            for i in range(0, len(class_array)):
                if class_array[i] == 0:
                    zeros.append(f[i])
                if class_array[i] == 1:
                    ones.append(f[i])
            feature_selection_[0][1].append(zeros)
            feature_selection_[1][1].append(ones)
        feature_selection_result = feature_selection(feature_selection_)
        for j in range(0, len(feature_selection_result)):
            if feature_selection_result[j] > 1:
                feature_columns.append(all_feature_columns[j])
                selected_column_indexes.append(j + 1)
    else:
        if optimize_binary:
            feature_columns = []
            for c in all_feature_columns:
                f = df[c].values
                gain = get_maximum_information_gain_for_categorical_features(f, class_array)
                if gain > 0.1:
                    feature_columns.append(c)
        else:
            feature_columns = all_feature_columns
    if use_boxplot:
        boxplot = df.boxplot(return_type='dict')
        for k in range(0, len(boxplot['fliers'])):
            if k in selected_column_indexes:
                mean = round(sum(df[columns[k]]) / len(df[columns[k]]))
                outliers = boxplot['fliers'][k]._y
                for l in range(0, len(df[columns[k]])):
                    if df[columns[k]][l] in outliers:
                        df[columns[k]][l] = mean
    all_samples = df[feature_columns].values
    model_.fit(all_samples, class_array)
    predict_array = df_test[feature_columns].values
    predicted = model_.predict(predict_array)
    total_true_predict = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0
    for i in range(0, len(test_class_array)):
        if test_class_array[i] == predicted[i]:
            total_true_predict = total_true_predict + 1
            if test_class_array[i] == 1:
                true_positive = true_positive + 1
            if test_class_array[i] == 0:
                true_negative = true_negative + 1
        else:
            if test_class_array[i] == 1:
                false_positive = false_positive + 1
            if test_class_array[i] == 0:
                false_negative = false_negative + 1
    confusion_matrix = [[], []]
    confusion_matrix[0].append(true_positive)
    confusion_matrix[0].append(false_negative)
    confusion_matrix[1].append(false_positive)
    confusion_matrix[1].append(true_negative)
    print('total test sample count : ', len(test_class_array))
    print('true prediction count : ', total_true_predict)
    print('accuracy : ', (total_true_predict / len(test_class_array)))
    print('confusion matrix [[TP,FN],[FP,TN]')
    print(confusion_matrix[0])
    print(confusion_matrix[1])


print('------------BAYES---------')
model = GaussianNB()
print('analyzing numerical data without optimization')
process(model, 'SPECTF-train.csv', 'SPECTF-test.csv', optimize_binary=False, optimize_numerical=False)
print('analyzing numerical data with feature selection')
process(model, 'SPECTF-train.csv', 'SPECTF-test.csv', optimize_binary=False, optimize_numerical=True)
print('analyzing numerical data with outlier analyze')
process(model, 'SPECTF-train.csv', 'SPECTF-test.csv', optimize_binary=False, optimize_numerical=True, use_boxplot=True)
model2 = CategoricalNB()
print('analyzing binary data without optimization')
process(model2, 'SPECT-train.csv', 'SPECT-test.csv', optimize_binary=False, optimize_numerical=False)
print('analyzing binary data with optimization')
process(model2, 'SPECT-train.csv', 'SPECT-test.csv', optimize_binary=True, optimize_numerical=False)
