import numpy as np
def data_processing(file_name):
    data_path = "data/" + file_name
    data = open(data_path)
    feature_names = data.readline().strip("\n").split(",")
    feature_vars = [[] for i in range(len(feature_names))]
    num_examples = 0

    end = False
    while not end:
        line = data.readline().strip("\n").split(",")
        if line == ['']:
            end = True
            data.close()
        else:
            for i, f in enumerate(line):
                if f not in feature_vars[i]:
                    feature_vars[i].append(f)
            num_examples += 1
    num_features = 0
    num_class = len(feature_vars[-1])
    for i in range(len(feature_names) - 1):
        c = len(feature_vars[i]) if len(feature_vars[i]) > 2 else 1
        num_features += c

    data_features = np.zeros(shape=(num_examples, num_features), dtype=np.int)
    data_classes = np.zeros(shape=(num_examples, num_class), dtype=np.int)

    data = open(data_path)
    data.readline()  # Skip the header
    end = False
    curr_exmaple_index = 0

    while not end:
        line = data.readline().strip("\n").split(",")
        if line == ['']:
            end = True
            data.close()
        else:
            for i, f in enumerate(line[:-1]):
                num_prev_vars = 0
                for j in range(i):
                    num_prev_vars += len(feature_vars[j]) if len(feature_vars[j]) > 2 else 1

                if len(feature_vars[i]) > 2:
                    curr_f_index = feature_vars[i].index(f) + num_prev_vars
                    data_features[curr_exmaple_index, curr_f_index] = 1
                else:
                    curr_f_index = num_prev_vars
                    if curr_exmaple_index == 0:
                        first_line = line
                        data_features[0, curr_f_index] = 0 if str(first_line[i]).strip().upper() == 'FALSE' or str(
                            first_line[i]).strip() == '0' \
                                                              or str(first_line[i]).strip().upper() == 'NO' or str(
                            first_line[i]).strip().upper() == 'WEAK' \
                                                              or str(first_line[i]).strip().upper() == 'NORMAL' else 1
                    else:
                        data_features[curr_exmaple_index, curr_f_index] = data_features[0, curr_f_index] if line[i] == \
                                                                                                            first_line[
                                                                                                                i] \
                            else (data_features[0, curr_f_index] + 1) % 2

            data_classes[curr_exmaple_index][feature_vars[-1].index(line[-1])] = 1
            curr_exmaple_index += 1

    return feature_names, feature_vars, data_features, data_classes, num_class