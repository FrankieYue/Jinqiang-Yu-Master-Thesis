from util.WholeMaxSATdssolver import WholeMaxSATdsSolver
from util.WholeMaxSATds import WholeMaxSATds
from util.dataProcessing import data_processing
import statistics
import os

def WholeMaxSATDSSolve(filename, no_ins, N):
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

    for ins in range(no_ins):
        print("Test {0}:".format(ins + 1))
        sols = []
        num_nodes = []
        num_sol = 0
        num_literal = []

        clauses = []
        variables = []
        all_runtime = []
        solved_runtime = []
        accu_runtime = 0

        found = False

        found = False
        while not found:
            DDS_solver = WholeMaxSATdsSolver(K, N, data)

            solutions = DDS_solver.solve()
            all_runtime.append(DDS_solver.runtime)
            accu_runtime += DDS_solver.runtime

            if solutions is not None:
                found = True
                solved_runtime.append(DDS_solver.runtime)
                for sol in solutions:
                    sols.append(sol)
                    num_nodes.append(N)

                    ds = WholeMaxSATds(K, N)
                    ds.parse_solution(sol)
                    ds.generate_tree()
                    ds.validate(data_features, data_classes)

                    lit = ds.get_num_literals()
                    num_literal.append(lit)
                    print("\nThe current number of nodes are: {0}".format(N) + ". Solution found")
                    print("The number of nodes is {0}. Literals: {1}, Rules: {2}".format(ds.valid_N, lit,
                                                                                         ds.valid_N - lit))
                    print("Hard clause: {0}, soft clause: {1}, variables: {2}".format(len(DDS_solver.cnf.hard),
                                                                                      len(DDS_solver.cnf.soft),
                                                                                      DDS_solver.cnf.nv))
                    print('Runtime: {0}s'.format('%.2f' % DDS_solver.runtime))
                    print('Accumulated runtime: {0}s'.format('%.2f' % accu_runtime))
                    try:
                        DDS_solver.cnf.to_file('trees/' + filename[:-4] + '/whole_model.cnf')
                    except FileNotFoundError as e:
                        os.makedirs('trees/' + filename[:-4])

                    clauses.append(len(DDS_solver.cnf.hard))
                    variables.append(DDS_solver.cnf.nv)
                    ds.draw(num_sol, None, filename[:-4])

            else:
                print("\nThe current number of nodes are: {0}".format(N) + ". Solution not found")
                print("Clause: {0}, variables: {1}".format(len(DDS_solver.cnf.hard), DDS_solver.cnf.nv))
                print('Runtime: {0}s'.format('%.2f' % DDS_solver.runtime))
                print('Accumulated runtime: {0}s'.format('%.2f' % accu_runtime))
                N += 1
                assert accu_runtime <= 3600, 'Runtime over 1h'




