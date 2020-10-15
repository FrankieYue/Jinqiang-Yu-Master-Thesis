from util.SplitMaxSATdssolver import SplitMaxSATdsSolver
from util.SplitMaxSATds import SplitMaxSATds
from util.dataProcessing import data_processing
import statistics
import os

def SplitMaxSATDSSolve(filename, no_ins, N):
    print("File: {0}\n".format(filename))
    feature_names, feature_vars, data_features, data_classes, num_class = data_processing(filename)

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

    for i in range(len(data_features)):
        for j in range(i + 1, len(data_features)):
            if list(data_features[i]) == list(data_features[j]):
                if list(data_classes[i]) != list(data_classes[j]):
                    data_classes[i] = most_class
                    data_classes[j] = most_class
                    assert list(data_classes[i]) == list(data_classes[j]), "Overlap! Index: {0}, {1}".format(i, j)

    data = (data_features, data_classes)
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
        variables = []
        all_runtime = [[] for i in range(num_class)]
        solved_runtime = []
        accu_runtime = 0

        for class_ids in range(num_class):
            found = False

            found = False
            while not found:
                DDS_solver = SplitMaxSATdsSolver(K, N, data[class_ids])

                solutions = DDS_solver.solve()
                all_runtime[class_ids].append(DDS_solver.runtime)
                accu_runtime += DDS_solver.runtime

                if solutions is not None:
                    found = True
                    solved_runtime.append(DDS_solver.runtime)
                    for sol in solutions:
                        sols.append(sol)
                        num_nodes.append(N)

                        ds = SplitMaxSATds(K, N)
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
                            DDS_solver.cnf.to_file('trees/' + filename[:-4] + '/' + str(class_ids) + '.cnf')
                        except FileNotFoundError as e:
                            os.makedirs('trees/' + filename[:-4])

                        clauses.append(len(DDS_solver.cnf.hard))
                        variables.append(DDS_solver.cnf.nv)
                        ds.draw(num_sol, class_ids, filename[:-4])
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

        ds = SplitMaxSATds(K, sum(num_nodes))
        ds.com_draw(filename[:-4], sols, num_valid_nodes)
        ds.com_validate(data_features, data_classes, num_valid_nodes, num_class)

        for i in range(len(clauses)):
            print("For class {0}, clauses: {1}, variables {2}".format(i + 1, clauses[i], variables[i]))

        print("Total clause: {0}, variables: {1} \n".format(sum(clauses), sum(variables)))

        for i in range(len(num_nodes)):
            print(
                "For class{0}, the number of nodes: {1}, the number of literals: {2}, the number of rules: {3}".format(
                    i + 1, num_valid_nodes[i], num_literal[i], num_valid_nodes[i] - num_literal[i]))

        print("The total number of nodes: {0}".format(sum(num_valid_nodes)))
        print("The total number of literals: {0}, the total number of rules: {1} \n".format(sum(num_literal),
                                                                                            sum(num_valid_nodes) - sum(
                                                                                                num_literal)))

        a_rt = 0
        for i in range(len(all_runtime)):
            a_rt += sum(all_runtime[i])
            print("For class {0}, the total runtime is {1}s".format(i + 1, '%.2f' % sum(all_runtime[i])))

        print('ALL runtime: {0}s \n'.format('%.2f' % a_rt))

        for i in range(len(solved_runtime)):
            print("For class {0}, the solved runtime is {1}s".format(i + 1, '%.2f' % solved_runtime[i]))

        print('Solved-model runtime: {0}s \n'.format('%.2f' % sum(solved_runtime)))

        ins_all_runtime.append(all_runtime)
        ins_solved_runtime.append(solved_runtime)

    print("Summary:")

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