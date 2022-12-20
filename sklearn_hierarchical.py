from sklearn.cluster import AgglomerativeClustering
import pandas as pd


def process(train_file_name):
    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    df = pd.read_csv(train_file_name)
    columns = df.columns.values
    class_column = columns[0]
    feature_columns = columns[1: len(columns)]
    class_array = df[class_column]
    all_samples = df[feature_columns].values
    model.fit(all_samples)
    total_true_predict = 0
    total_true_predict2 = 0
    for i in range(0, len(model.labels_)):
        if model.labels_[i] == class_array[i]:
            total_true_predict = total_true_predict + 1
        if model.labels_[i] == 0 and class_array[i] == 1:
            total_true_predict2 = total_true_predict2 + 1
        if model.labels_[i] == 1 and class_array[i] == 0:
            total_true_predict2 = total_true_predict2 + 1

    if total_true_predict > total_true_predict2:
        print('calculate label prediction normally')
        print('total test sample count : ', len(class_array))
        print('true prediction count : ', total_true_predict)
        print('accuracy : ', (total_true_predict / len(class_array)))
    else:
        print('calculate exchanged label prediction')
        print('total test sample count : ', len(class_array))
        print('true prediction count : ', total_true_predict2)
        print('accuracy : ', (total_true_predict2 / len(class_array)))


print('-------------BISECTING K-MEANS---------')
print('numerical data set')
process('SPECTF-train.csv')
print('binary data set')
process('SPECT-train.csv')
