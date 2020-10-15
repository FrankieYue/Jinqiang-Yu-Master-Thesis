from util.SepSparseMaxSATdssolver import SepSparseMaxSATdsSolver
from util.SepSparseMaxSATds import SepSparseMaxSATds
from util.dataProcessing import data_processing
import statistics
import os

def SepSparseMaxSATDSSolve(train_filename, test_filename, no_ins, start_nodes, lam, mis_weight=1):
    print("File: {0}\n".format(train_filename))
    print("File: {0}\n".format(test_filename))
    feature_names, feature_vars, data_features, data_classes, num_class = data_processing(train_filename)

    print(feature_names)
    print(feature_vars)
    # print("features:\n", data_features)
    # print("\n classes:\n", data_classes)
    print("num of data:", len(data_features))
    print("num of features:", len(data_features[0]))
    print("num of classes:", num_class)

    K = data_features.shape[1]

    class_count = [0 for i in range(num_class)]
    for i in range(len(data_classes)):
        for j in range(len(data_classes[i])):
            if data_classes[i][j] == 1:
                class_count[j] += 1
                continue

    most_class = [0 for i in range(num_class)]
    most_class[class_count.index(max(class_count))] = 1

    '''
    #Deal with inconsistent dataset
    for i in range(len(data_features)):
        for j in range(i + 1, len(data_features)):
            if list(data_features[i]) == list(data_features[j]):
                if list(data_classes[i]) != list(data_classes[j]):
                    data_classes[i] = most_class
                    data_classes[j] = most_class
                    assert list(data_classes[i]) == list(data_classes[j]), "Overlap! Index: {0}, {1}".format(i, j)
    '''

    ins_all_runtime = []
    ins_solved_runtime = []
    all_data_feature = [[] for i in range(num_class)]
    for i in range(len(data_features)):
        for j in range(num_class):
            if data_classes[i][j] == 1:
                all_data_feature[j].append(list(data_features[i]))
    data = []
    for i in range(num_class):
        selected_features = []
        non_selected_features = []
        for j in range(num_class):
            if i != j:
                non_selected_features += all_data_feature[j]
            else:
                selected_features.append(all_data_feature[j])

        data.append((selected_features[0], non_selected_features))

    for ins in range(no_ins):
        print("Test {0}:".format(ins + 1))
        num_sol = 1
        sols = []
        num_nodes = []
        num_valid_nodes = []
        num_literal = []

        clauses = []
        soft_clauses =[]
        variables = []
        all_runtime = [[] for i in range(num_class)]
        solved_runtime = []
        accu_runtime = 0

        for class_ids in range(num_class):
            N = start_nodes

            found = False
            while not found:
                DDS_solver = SepSparseMaxSATdsSolver(K, N, data[class_ids], lam, mis_weight)

                solutions = DDS_solver.solve()
                all_runtime[class_ids].append(DDS_solver.runtime)
                accu_runtime += DDS_solver.runtime

                if solutions is not None:
                    found = True
                    solved_runtime.append(DDS_solver.runtime)
                    for sol in solutions:
                        sols.append(sol)
                        num_nodes.append(N)

                        ds = SepSparseMaxSATds(K, N)
                        ds.parse_solution(sol)
                        ds.generate_tree()

                        lit = ds.get_num_literals()

                        num_valid_nodes.append(ds.valid_N)
                        num_literal.append(lit)
                        print("\nFor class {0}, the current number of nodes are: {1}".format(class_ids + 1,
                                                                                             N) + ". Solution found")
                        print("The number of nodes is {0}. Literals: {1}, Rules: {2}".format(ds.valid_N, lit,
                                                                                             ds.valid_N - lit))
                        #   print("Clause: {0}, variables: {1}".format(len(DDS_solver.cnf.clauses), DDS_solver.cnf.nv))
                        print("Hard clause: {0}, soft clause: {1}, variables: {2}".format(len(DDS_solver.cnf.hard),
                                                                                          len(DDS_solver.cnf.soft),
                                                                                          DDS_solver.cnf.nv))
                        print('Runtime: {0}s'.format('%.2f' % DDS_solver.runtime))
                        print('Accumulated runtime: {0}s'.format('%.2f' % accu_runtime))
                        try:
                            DDS_solver.cnf.to_file('trees/' + train_filename[:-4] + '/' + str(class_ids) + '.cnf')
                        except FileNotFoundError as e:
                            os.makedirs('trees/' + train_filename[:-4])

                        clauses.append(len(DDS_solver.cnf.hard))
                        soft_clauses.append(len(DDS_solver.cnf.soft))
                        variables.append(DDS_solver.cnf.nv)
                        ds.draw(num_sol, class_ids, train_filename[:-4])
                        num_sol += 1
                        if num_sol > 30:
                            break
                else:
                    print("\nFor class {0}, the current number of nodes are: {1}".format(class_ids + 1,
                                                                                         N) + ". Solution not found")
                    # print("Clause: {0}, variables: {1}".format(len(DDS_solver.cnf.clauses), DDS_solver.cnf.nv))
                    print("Hard clause: {0}, soft clause: {1}, variables: {2}".format(len(DDS_solver.cnf.hard),
                                                                                      len(DDS_solver.cnf.soft),
                                                                                      DDS_solver.cnf.nv))
                    print('Runtime: {0}s'.format('%.2f' % DDS_solver.runtime))
                    print('Accumulated runtime: {0}s'.format('%.2f' % accu_runtime))
                    N += 1
                    assert accu_runtime <= 3600, 'Runtime over 1h'

        ds = SepSparseMaxSATds(K, sum(num_nodes))
        ds.com_draw(train_filename[:-4], sols, num_valid_nodes)
        ds.com_validate(data_features, data_classes, num_valid_nodes, num_class)
        print('\nFor training datasets, the number of misclassification: {0}, accuracy: {1}'.format(ds.mis_count, '%.2f'%((1 - ds.mis_count / len(data_classes)) * 100) + '%'))

        test_feature_names, test_feature_vars, test_data_features, test_data_classes, test_num_class = data_processing(test_filename)
        ds.mis_count = 0
        print(ds.mis_count)
        ds.com_validate(test_data_features, test_data_classes, num_valid_nodes, test_num_class)
        print(ds.mis_count)
        print('\nFor testing datasets, the number of misclassification: {0}, accuracy: {1}'.format(ds.mis_count, '%.2f' % ((1 - ds.mis_count / len(test_data_classes)) * 100) + '%'))


        count_hard_clause = str(sum(clauses)) + ": "
        count_soft_clause = str(sum(soft_clauses)) + ": "
        count_variable = str(sum(variables)) + ": "

        for i in range(len(clauses)):
            count_hard_clause = count_hard_clause + str(clauses[i]) + ' + ' if i < (len(clauses)-1) else count_hard_clause + str(clauses[i])
            count_soft_clause = count_soft_clause + str(soft_clauses[i]) + ' + ' if i < (len(clauses) - 1) else count_soft_clause + str(soft_clauses[i])
            count_variable = count_variable + str(variables[i]) + ' + ' if i < (len(clauses) - 1) else count_variable + str(variables[i])
            #print("For class {0}, hard clauses: {1}, soft clause: {2}, variables {3}".format(i + 1, clauses[i], soft_clauses[i], variables[i]))

        print("Total hard clauses: {0}".format(count_hard_clause))
        print("Total soft clauses: {0}".format(count_soft_clause))
        print("Total variables: {0}".format(count_variable))

        count_node = str(sum(num_valid_nodes)) + ": "
        count_literal = str(sum(num_literal)) + ": "
        count_rule = str(sum(num_valid_nodes) - sum(num_literal)) + ": "
        for i in range(len(num_nodes)):
            count_node = count_node + str(num_valid_nodes[i]) + ' + ' if i < (len(num_nodes) - 1) else count_node + str(num_valid_nodes[i])
            count_literal = count_literal + str(num_literal[i]) + ' + ' if i < (len(num_nodes) - 1) else count_literal + str(num_literal[i])
            count_rule = count_rule + str(num_valid_nodes[i] - num_literal[i]) + ' + ' if i < ( len(num_nodes) - 1) else count_rule + str(num_valid_nodes[i] - num_literal[i])

           # print("For class{0}, the number of nodes: {1}, the number of literals: {2}, the number of rules: {3}".format(i + 1, num_valid_nodes[i], num_literal[i], num_valid_nodes[i] - num_literal[i]))

        print("\nThe total number of nodes: {0}".format(count_node))
        print("The total number of literals: {0}".format(count_literal))
        print("The total number of rules: {0} \n".format(count_rule))

        '''
        a_rt = 0
        for i in range(len(all_runtime)):
            a_rt += sum(all_runtime[i])
            print("For class {0}, the total runtime is {1}s".format(i + 1, '%.2f' % sum(all_runtime[i])))

        print('ALL runtime: {0}s \n'.format('%.2f' % a_rt))
        '''
        sum_runtime = '%.2f' % sum(solved_runtime) + ": "
        for i in range(len(solved_runtime)):
            sum_runtime = sum_runtime + '%.2f' %(solved_runtime[i]) + ' + ' if i < (len(solved_runtime) - 1) else sum_runtime + '%.2f' %(solved_runtime[i])
            #print("For class {0}, the solved runtime is {1}s".format(i + 1, '%.2f' % solved_runtime[i]))

        print('Solved-model runtime: {0}'.format(sum_runtime))

        ins_all_runtime.append(all_runtime)
        ins_solved_runtime.append(solved_runtime)

    if no_ins >1 :
        print("\nSummary:")

        class_total_time = [[] for i in range(num_class)]

        for i in range(num_class):
            for j in range(len(ins_all_runtime)):
                class_total_time[i].append(sum(ins_all_runtime[j][i]))

        avg_total_time = []
        for i in range(num_class):
            print(
                'For class {0}, the average total time: {1}'.format(i + 1, "%0.2f" % statistics.mean(class_total_time[i])))
            avg_total_time.append(statistics.mean(class_total_time[i]))

        print("All average total runtime: {0}\n".format("%0.2f" % sum(avg_total_time)))

        class_sol_time = [[] for i in range(num_class)]
        for i in range(num_class):
            for j in range(len(ins_solved_runtime)):
                class_sol_time[i].append(ins_solved_runtime[j][i])

        avg_sol_time = []
        for i in range(num_class):
            print('For class {0}, the average solved time: {1}'.format(i + 1, "%0.2f" % statistics.mean(class_sol_time[i])))
            avg_sol_time.append(statistics.mean(class_sol_time[i]))

        print("All average solved time: {0}\n".format("%0.2f" % sum(avg_sol_time)))