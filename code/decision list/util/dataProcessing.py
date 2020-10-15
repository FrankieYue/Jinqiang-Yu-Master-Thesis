import gzip
def data_processing(file_name, test_filename=None):
    data = gzip.open(file_name, 'rt')

    feature_names = data.readline().strip("\n").split(",")
    feature_vars = [[] for i in range(len(feature_names))]
    num_examples = 0

    if test_filename is not None:
        test_data = gzip.open(test_filename, 'rt')

        test_data.readline().strip("\n").split(",")
        test_num_examples = 0

    end = False
    while not end:
        line = data.readline().strip("\n").split(",")
        if line == ['']:
            end = True
            data.close()
        else:
            for i, f in enumerate(line):
                if f not in feature_vars[i]:
                    if str(f).upper() in ('TRUE', '1', 'YES', 'STRONG'):
                        feature_vars[i].insert(0, f)
                    else:
                        feature_vars[i].append(f)
            num_examples += 1

    if test_filename is not None:
        end = False
        while not end:
            line = test_data.readline().strip("\n").split(",")
            if line == ['']:
                end = True
                test_data.close()
            else:
                for i, f in enumerate(line):
                    if f not in feature_vars[i]:
                        feature_vars[i].append(f)
                test_num_examples += 1

    num_features = 0
    num_class = len(feature_vars[-1])
    for i in range(len(feature_names) - 1):
        c = len(feature_vars[i]) if len(feature_vars[i]) > 2 else 1
        num_features += c

    data_features = [ [0 for j in range(num_features)] for i in range(num_examples)]
    data_classes =[ [0 for j in range(num_class)] for i in range(num_examples)]

    data = gzip.open(file_name, 'rt')
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
                        data_features[0][curr_f_index] = 0 if str(first_line[i]).strip().upper() in ('FALSE', '0', 'NO', 'WEAK', 'NORMAL') else 1
                    else:
                        data_features[curr_exmaple_index][curr_f_index] = data_features[0][curr_f_index] if line[i] == first_line[i] else (data_features[0][curr_f_index] + 1) % 2

            data_classes[curr_exmaple_index][feature_vars[-1].index(line[-1])] = 1
            curr_exmaple_index += 1

    if test_filename is not None:
        test_data_features = [[0 for j in range(num_features)] for i in range(test_num_examples)]
        test_data_classes = [[0 for j in range(num_class)] for i in range(test_num_examples)]

        test_data = gzip.open(test_filename, 'rt')
        test_data.readline()  # Skip the header
        end = False
        curr_exmaple_index = 0

        while not end:
            line = test_data.readline().strip("\n").split(",")
            if line == ['']:
                end = True
                test_data.close()
            else:
                for i, f in enumerate(line[:-1]):
                    num_prev_vars = 0
                    for j in range(i):
                        num_prev_vars += len(feature_vars[j]) if len(feature_vars[j]) > 2 else 1

                    if len(feature_vars[i]) > 2:
                        curr_f_index = feature_vars[i].index(f) + num_prev_vars
                        test_data_features[curr_exmaple_index][curr_f_index] = 1
                    else:
                        curr_f_index = num_prev_vars
                        test_data_features[curr_exmaple_index][curr_f_index] = data_features[0][curr_f_index] if line[i] == first_line[i] else (data_features[0][curr_f_index] + 1) % 2

                test_data_classes[curr_exmaple_index][feature_vars[-1].index(line[-1])] = 1
                curr_exmaple_index += 1


    if test_filename == None:
        return feature_names, feature_vars, data_features, data_classes, num_class
    else:
        return feature_names, feature_vars, data_features, data_classes, num_class, test_data_features, test_data_classes