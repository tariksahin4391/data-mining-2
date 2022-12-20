import math
import pandas as pd


class FeatureAndValue:
    feature_value = -1
    class_value = -1
    index = -1

    def __init__(self, feature_value, class_value, index):
        self.feature_value = feature_value
        self.class_value = class_value
        self.index = index


class FeatureAndValueCover:
    feature_name = ''
    feature_and_value_array = []

    def __init__(self, feature_name, arr):
        self.feature_name = feature_name
        self.feature_and_value_array = arr


class TreeNode:
    is_root = False
    feature_name = ''
    parent = None
    zero_subtree = None
    one_subtree = None
    zero_value = -1
    one_value = -1

    def __init__(self, is_root=False, feature_name='', parent=None, zero_subtree=None, one_subtree=None, zero_value=-1,
                 one_value=-1):
        self.is_root = is_root
        self.feature_name = feature_name
        self.parent = parent
        self.zero_subtree = zero_subtree
        self.one_subtree = one_subtree
        self.zero_value = zero_value
        self.one_value = one_value


binary_tree_root = TreeNode(is_root=True, feature_name='', parent=None)
features_and_values = []
binary_tree_root_2 = TreeNode(is_root=True, feature_name='', parent=None)
features_and_values_2 = []
purity = 0.9


# array içindeki elemanları gruplar
def group_array(arr):
    result_as_list = []
    for i in arr:
        if len(result_as_list) == 0:
            result_as_list.append([i, 1])
        else:
            found = False
            for e in result_as_list:
                if e[0] == i:
                    e[1] = e[1] + 1
                    found = True
                    break
            if not found:
                result_as_list.append([i, 1])
    result_as_tuple_list = []
    for r in result_as_list:
        result_as_tuple_list.append((r[0], r[1]))
    return result_as_tuple_list


# entropy hesaplayan fonksiyon
def calculate_entropy(decision_array):
    grouped = group_array(decision_array)
    total_elem_count = len(decision_array)
    total_entropy = 0
    for elem in grouped:
        total_entropy = total_entropy + (elem[1] / total_elem_count) * math.log2(elem[1] / total_elem_count)
    total_entropy = -1 * total_entropy
    return total_entropy


def get_maximum_information_gain_for_categorical_features(sample_arr1, decisions):
    general_entropy = calculate_entropy(decisions)
    total_elem_count = len(decisions)
    # kararların gruplanması
    grouped_samples = group_array(sample_arr1)  # sample arrayin kendi içinde gruplanması
    # I(1,3,2,...)
    local_gain = []
    grouped_decisions = group_array(decisions)
    for grouped_sample in grouped_samples:
        group_gain = []
        for grouped_decision in grouped_decisions:
            count = 0
            for j in range(0, len(decisions)):
                if sample_arr1[j] == grouped_sample[0] and decisions[j] == grouped_decision[0]:
                    count = count + 1
            group_gain.append(count)
        local_gain.append(group_gain)
    total = 0
    for elem in local_gain:  # [3,2]
        elem_gain_total = 0
        elem_total = sum(elem)
        for e in elem:
            if not e == 0:
                elem_gain_total = elem_gain_total + ((e / elem_total) * math.log2((e / elem_total)))
        elem_gain_total = elem_gain_total * -1
        elem_gain_total = (elem_total / total_elem_count) * elem_gain_total
        total = total + elem_gain_total
    gain = general_entropy - total
    return gain


def generate_tree(f: FeatureAndValueCover, parent=None, step=-1, global_root=None, from_zero=False, optimize=False):
    if optimize:
        features_and_values_ = features_and_values_2
    else:
        features_and_values_ = features_and_values
    if global_root is None:
        this_node = TreeNode(is_root=False, feature_name=f.feature_name, parent=parent)
        if from_zero:
            parent.zero_subtree = this_node
        else:
            parent.one_subtree = this_node
    else:
        this_node = global_root
    zeros = [fv for fv in f.feature_and_value_array if fv.feature_value == 0]
    ones = [fv for fv in f.feature_and_value_array if fv.feature_value == 1]
    decisions_zero_for_zeros = [fv for fv in zeros if fv.class_value == 0]
    decisions_one_for_zeros = [fv for fv in zeros if fv.class_value == 1]
    decisions_zero_for_ones = [fv for fv in ones if fv.class_value == 0]
    decisions_one_for_ones = [fv for fv in ones if fv.class_value == 1]
    if step == 20:
        # print('last step')
        if len(decisions_zero_for_zeros) > len(decisions_one_for_zeros):
            this_node.zero_value = 0
        else:
            this_node.zero_value = 1
        if len(decisions_zero_for_ones) > len(decisions_one_for_ones):
            this_node.one_value = 0
        else:
            this_node.one_value = 1
    else:
        # print('0 için saflığı hesapla')
        if len(zeros) == 0 or len(decisions_zero_for_zeros) == 0 or len(decisions_one_for_zeros) == 0 or len(
                decisions_zero_for_zeros) / len(zeros) >= purity or len(decisions_one_for_zeros) / len(zeros) >= purity:
            if len(zeros) == 0:
                this_node.zero_value = -1
            else:
                if len(decisions_zero_for_zeros) > len(decisions_one_for_zeros):
                    this_node.zero_value = 0
                else:
                    this_node.zero_value = 1
            # print('0 için saflığa ulaşıldı')
        else:
            filtered_fvs = []
            for fv in features_and_values_:
                if fv.feature_name != f.feature_name:
                    filtered_arr = [flt for flt in fv.feature_and_value_array if
                                    flt.index in list(map(lambda x: x.index, zeros))]
                    add = FeatureAndValueCover(fv.feature_name, filtered_arr)
                    filtered_fvs.append(add)
            ftr_and_gains = []
            for to_gain in filtered_fvs:
                sample_arr1 = list(map(lambda x: x.feature_value, to_gain.feature_and_value_array))
                decision_arr1 = list(map(lambda x: x.class_value, to_gain.feature_and_value_array))
                gain1 = get_maximum_information_gain_for_categorical_features(sample_arr1, decision_arr1)
                ftr_and_gains.append([to_gain.feature_name, gain1])
            ftr_and_gains.sort(key=lambda x: x[1], reverse=True)
            root_val1 = [f1 for f1 in filtered_fvs if f1.feature_name == ftr_and_gains[0][0]][0]
            generate_tree(root_val1, this_node, step + 1, global_root=None, from_zero=True, optimize=optimize)
        # print('1 için saflığı hesapla')
        if len(ones) == 0 or len(decisions_zero_for_ones) == 0 or len(decisions_one_for_ones) == 0 or len(
                decisions_zero_for_ones) / len(ones) >= purity or len(decisions_one_for_ones) / len(ones) >= purity:
            if len(ones) == 0:
                this_node.one_value = -1
            else:
                if len(decisions_zero_for_ones) > len(decisions_one_for_ones):
                    this_node.one_value = 0
                else:
                    this_node.one_value = 1
            # print('1 için saflığa ulaşıldı')
        else:
            filtered_fvs = []
            for fv in features_and_values_:
                if fv.feature_name != f.feature_name:
                    filtered_arr = [flt for flt in fv.feature_and_value_array if
                                    flt.index in list(map(lambda x: x.index, ones))]
                    add = FeatureAndValueCover(fv.feature_name, filtered_arr)
                    filtered_fvs.append(add)
            ftr_and_gains = []
            for to_gain in filtered_fvs:
                sample_arr1 = list(map(lambda x: x.feature_value, to_gain.feature_and_value_array))
                decision_arr1 = list(map(lambda x: x.class_value, to_gain.feature_and_value_array))
                gain1 = get_maximum_information_gain_for_categorical_features(sample_arr1, decision_arr1)
                ftr_and_gains.append([to_gain.feature_name, gain1])
            ftr_and_gains.sort(key=lambda x: x[1], reverse=True)
            root_val1 = [f1 for f1 in filtered_fvs if f1.feature_name == ftr_and_gains[0][0]][0]
            generate_tree(root_val1, this_node, step + 1, global_root=None, from_zero=False, optimize=optimize)


df_binary = pd.read_csv('SPECT-train.csv')
df_test = pd.read_csv('SPECT-test.csv')
binary_columns = df_binary.columns.values
class_column = binary_columns[0]
feature_columns = binary_columns[1: len(binary_columns)]
reduced_feature_columns = []
class_array = df_binary[class_column]
test_class_array = df_test[class_column]
# binary_samples = df_binary.values
# boxplot = df_binary.boxplot()
for f in feature_columns:
    values = df_binary[f].values
    feature_and_value_cover = FeatureAndValueCover(f, [])
    for i in range(0, len(values)):
        feature_and_value = FeatureAndValue(values[i], class_array[i], i)
        feature_and_value_cover.feature_and_value_array.append(feature_and_value)
    features_and_values.append(feature_and_value_cover)

features_and_gains = []
for f in features_and_values:
    sample_arr = list(map(lambda x: x.feature_value, f.feature_and_value_array))
    decision_arr = list(map(lambda x: x.class_value, f.feature_and_value_array))
    gain = get_maximum_information_gain_for_categorical_features(sample_arr, decision_arr)
    if gain > 0.05:
        features_and_values_2.append(f)
        reduced_feature_columns.append(f.feature_name)
    features_and_gains.append([f.feature_name, gain])

features_and_gains.sort(key=lambda x: x[1], reverse=True)
print(features_and_gains)
root_val = [f for f in features_and_values if f.feature_name == features_and_gains[0][0]][0]
root_val_2 = [f for f in features_and_values if f.feature_name == features_and_gains[0][0]][0]
binary_tree_root.feature_name = root_val.feature_name
generate_tree(root_val, None, 1, binary_tree_root)
binary_tree_root_2.feature_name = root_val_2.feature_name
generate_tree(root_val_2, None, 1, binary_tree_root_2, optimize=True)


def predict(fn_and_sample, node: TreeNode):
    which_feature = []
    for fn in fn_and_sample:
        if fn[0] == node.feature_name:
            which_feature = fn
            break
    if which_feature[1] == 0:
        if node.zero_subtree is not None:
            return predict(fn_and_sample, node.zero_subtree)
        else:
            return node.zero_value
    else:
        if node.one_subtree is not None:
            return predict(fn_and_sample, node.one_subtree)
        else:
            return node.one_value


total_true_count = 0
true_positive = 0
false_negative = 0
false_positive = 0
true_negative = 0
test_samples = df_test[feature_columns].values
for i2 in range(0, len(test_class_array)):
    test_sample = test_samples[i2]
    feature_name_and_sample = []
    for i1 in range(0, len(feature_columns)):
        feature_name_and_sample.append([feature_columns[i1], test_sample[i1]])
    result = predict(feature_name_and_sample, binary_tree_root)
    if result == test_class_array[i2]:
        total_true_count = total_true_count + 1
        if test_class_array[i2] == 1:
            true_positive = true_positive + 1
        if test_class_array[i2] == 0:
            true_negative = true_negative + 1
    else:
        if test_class_array[i2] == 1:
            false_positive = false_positive + 1
        if test_class_array[i2] == 0:
            false_negative = false_negative + 1

confusion_matrix = [[], []]
confusion_matrix[0].append(true_positive)
confusion_matrix[0].append(false_negative)
confusion_matrix[1].append(false_positive)
confusion_matrix[1].append(true_negative)
print('analyzing binary data without optimization')
print('total test sample count : ', len(test_class_array))
print('true prediction count : ', total_true_count)
print('accuracy : ', (total_true_count / len(test_class_array)))
print('confusion matrix [[TP,FN],[FP,TN]')
print(confusion_matrix[0])
print(confusion_matrix[1])

################## herhangi optimizasyon yapılmadan ###############

total_true_count = 0
true_positive = 0
false_negative = 0
false_positive = 0
true_negative = 0
test_samples = df_test[reduced_feature_columns].values
for i2 in range(0, len(test_class_array)):
    test_sample = test_samples[i2]
    feature_name_and_sample = []
    for i1 in range(0, len(reduced_feature_columns)):
        feature_name_and_sample.append([reduced_feature_columns[i1], test_sample[i1]])
    result = predict(feature_name_and_sample, binary_tree_root_2)
    if result == test_class_array[i2]:
        total_true_count = total_true_count + 1
        if test_class_array[i2] == 1:
            true_positive = true_positive + 1
        if test_class_array[i2] == 0:
            true_negative = true_negative + 1
    else:
        if test_class_array[i2] == 1:
            false_positive = false_positive + 1
        if test_class_array[i2] == 0:
            false_negative = false_negative + 1

confusion_matrix = [[], []]
confusion_matrix[0].append(true_positive)
confusion_matrix[0].append(false_negative)
confusion_matrix[1].append(false_positive)
confusion_matrix[1].append(true_negative)
print('analyzing binary data with optimization')
print('total test sample count : ', len(test_class_array))
print('true prediction count : ', total_true_count)
print('accuracy : ', (total_true_count / len(test_class_array)))
print('confusion matrix [[TP,FN],[FP,TN]')
print(confusion_matrix[0])
print(confusion_matrix[1])
