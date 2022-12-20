import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from decision_tree import get_maximum_information_gain_for_categorical_features
from sklearn_bayes import feature_selection


def process(train_file_name, test_file_name, optimize_binary=False, optimize_numerical=False, use_boxplot=False):
    df = pd.read_csv(train_file_name)
    df_test = pd.read_csv(test_file_name)
    columns = df.columns.values
    class_column = columns[0]
    all_feature_columns = columns[1: len(columns)]
    class_array = df[class_column]
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
    model = KNeighborsClassifier(n_neighbors=3)
    all_samples = df[feature_columns].values
    model.fit(all_samples, class_array)
    predict_array = df_test[feature_columns].values
    predicted = model.predict(predict_array)
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


print('-------KNN--------')
print('analyzing numerical data without optimization')
process('SPECTF-train.csv', 'SPECTF-test.csv')
print('analyzing numerical data with feature selection')
process('SPECTF-train.csv', 'SPECTF-test.csv', optimize_binary=False, optimize_numerical=True)
print('analyzing numrical data with outlier analyze')
process('SPECTF-train.csv', 'SPECTF-test.csv', optimize_binary=False, optimize_numerical=True, use_boxplot=True)
print('analyzing binary data without optimization')
process('SPECT-train.csv', 'SPECT-test.csv')
print('analyzing binary data with optimization')
process('SPECT-train.csv', 'SPECT-test.csv', optimize_binary=True, optimize_numerical=False)
