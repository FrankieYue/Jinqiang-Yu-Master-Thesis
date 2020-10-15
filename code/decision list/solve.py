from util.dlsolver import DLSolver
from util.dataProcessing import data_processing
from util.dl import DL
import math
import os
import itertools
import random

def Solve(train_filename, test_filename, num_nodes, lam, mis_weight=1, s_order='all', asc_order = 'desc', maxsat=True, sep=True, ml='list', sparse=False):
    print("File: {0}\n".format(train_filename))
    feature_names, feature_vars, data_features, data_classes, num_class, test_data_features, test_data_classes = data_processing(train_filename, test_filename)
    print(data_classes)
    #K = data_features.shape[1] # The number of features
    K = len(data_features[0])  # The number of features

    # To record the number of training items in each class
    class_count = [0 for i in range(num_class)]

    for i in range(len(data_classes)):
        for j in range(len(data_classes[i])):
            if data_classes[i][j] == 1:
                class_count[j] += 1
                continue

    most_class = [0 for i in range(num_class)]
    most_class[class_count.index(max(class_count))] = 1

    # only used for sparse models:
    LAM = math.ceil(lam * len(data_features))

    if not sparse:
        # If not sparse, deal with inconsistent dataset
        for i in range(len(data_features)):
            for j in range(i + 1, len(data_features)):
                if list(data_features[i]) == list(data_features[j]):
                    if list(data_classes[i]) != list(data_classes[j]):
                        data_classes[i] = most_class
                        data_classes[j] = most_class
                        assert list(data_classes[i]) == list(data_classes[j]), "Overlap! Index: {0}, {1}".format(i, j)

    # Separted Model
    if sep:

        start_node = num_nodes # The size of the model

        sep_order = s_order #Order type

        # Sort data based on their class
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

            data.append(selected_features[0])

        data_dict = {class_ids + 1: data[class_ids][:] for class_ids in range(len(data))}

        # The class order for separted models
        all_class_order = []
        if ml == 'list':

            if sep_order == 'all' and num_class <= 3:
                for c in itertools.permutations(range(1, num_class + 1), num_class):
                    all_class_order.append(list(c))
            elif sep_order not in ('accuy', 'cost'):
                if sep_order == 'maj':
                    # sep_order == 'maj'
                    # Class order in the number of items in each class
                    # E.g. class 1 has 4 items, class 2 has 6 items, the order would be [6, 4]
                    sort_c_count = [sorted(class_count).index(x) for x in class_count]
                    for ii in range(len(class_count)):
                        new = sort_c_count[:]
                        new.pop(ii)
                        while sort_c_count[ii] in new:
                            sort_c_count[ii] += 1

                    maj_order = [None for j in range(len(class_count))]

                    for ii in range(len(class_count)):
                        maj_order[len(class_count) - 1 - sort_c_count[ii]] = ii + 1
                    if asc_order == 'desc':
                        all_class_order = [maj_order]
                    elif asc_order == 'asc':
                        maj_order.reverse()
                        all_class_order = [maj_order]
                    else:
                        assert asc_order == 'both'
                        all_class_order.append(maj_order)
                        r_maj_order = maj_order[:]
                        r_maj_order.reverse()
                        all_class_order.append(r_maj_order)
                else:
                    assert sep_order == 'all' and num_class > 3
                    # Get 3 random orders
                    b_order = list(range(1, num_class + 1))
                    while True:
                        new_order = b_order[:]
                        random.shuffle(new_order)
                        if new_order not in all_class_order:
                            all_class_order.append(new_order)
                        if len(all_class_order) == 3:
                            break
                    assert len(all_class_order) == 3
        else:
            all_class_order.append(list(range(1, num_class+1)))

        all_order_detail = [] # Record the results of decision lists

        # The class order based on training accuracy
        # For step 1, the model goes through all classes respectively, and get the number of misclassification
        # The first decision list would be generated for the class with least misclassification
        # Repeat the step
        if sep_order in ('accuy', 'cost'):
            tem_order = list(range(1, num_class + 1)) # All possible classes for the first step
            for c_ids in range(num_class):
                all_class_order.append(tem_order[c_ids:] + tem_order[:c_ids])

            train_sol_rec = [{} for c_ids in range(num_class) ]
            train_sol_rec[0] = {c_ids + 1: {'mis_count': None, 'sol': None, 'best': False,
                                            'node': None, 'valid_node': None,'lit': None, 'time': 0,
                                            'data': None, 'cost': None}
                                for c_ids in range(num_class)}
            final_order = []

        done = False
        # For accuracy order only
        curr_pos = 0

        while not done:
            for class_order in all_class_order:
                if curr_pos == 0:
                    data = [data_dict[class_order[ii]][:] for ii in range(len(class_order))] # Data for the particular order
                elif sep_order in ('accuy', 'cost'):
                    data = [accu_new_data[class_order[ii]][:] for ii in range(len(class_order))]
                sols = [] # Record the solution for a particular order
                num_nodes = [] # Record the upper bound size
                num_valid_nodes = [] # Record the number of used nodes
                num_literal = [] # Number of literals

                # CNF clauses, variables
                clauses = []
                soft_clauses =[]
                variables = []

                # Record runtime
                all_runtime = [[] for i in range(num_class)]
                accu_runtime = 0

                # Upper bound for each class
                N = [start_node for i in range(num_class)]

                ubounds = [math.ceil((len(data_features[0]) + 1) * len(data[class_ids])) for class_ids in range(num_class)]

                # Go through all classes
                for class_ids in range(curr_pos, num_class):
                    found = False
                    solved = False
                    while not found:
                        DDL_solver = DLSolver(K, N[class_ids], data, class_ids=class_ids, LAM=LAM, mis_weight=mis_weight, maxsat=maxsat, sep=sep, ml=ml, sparse=sparse)
                        solutions = DDL_solver.solve()
                        all_runtime[class_ids].append(DDL_solver.runtime)
                        accu_runtime += DDL_solver.runtime

                        #print(N[class_ids], accu_runtime, datetime.now().strftime("%H:%M:%S"))
                        if solutions is not None:

                            for sol in solutions:

                                dl = DL(K, N[class_ids])
                                dl.parse_solution(sols=[sol])
                                dl.generate_node(cur_class=class_order[class_ids])

                                lit = dl.get_num_literals()
                                nof_miss = 0 #If not sparse


                                if sparse:
                                    # For upper bound
                                    nof_miss, misclas_ids = dl.get_miss(DDL_solver.M + DDL_solver.non_M, sol)
                                    nof_used = dl.valid_N
                                    cost = nof_used + int(math.ceil(nof_miss / LAM))
                                    ubounds[class_ids] = min(cost, ubounds[class_ids])
                                    if ubounds[class_ids] in (N[class_ids], nof_used):
                                        found = True
                                        solved = True
                                        sols.append(sol)

                                        filtered_data = DDL_solver.filtered_data[:]
                                        classified_data = DDL_solver.classified_data[:]

                                        if ml == 'list' and sparse: #Filter out classified data for the next step
                                            for ids in misclas_ids:
                                                for accu_num_ids in range(1, len(DDL_solver.num_data)):
                                                    if accu_num_ids == 1 and ids <= DDL_solver.num_data[accu_num_ids - 1]:
                                                        filtered_data[accu_num_ids - 1][ids - 1] = None
                                                        break
                                                    if ids <= DDL_solver.num_data[accu_num_ids] and ids > DDL_solver.num_data[accu_num_ids - 1]:
                                                        filtered_data[accu_num_ids][
                                                            ids - 1 - DDL_solver.num_data[accu_num_ids - 1]] = None
                                                        break

                                            for d in range(len(DDL_solver.filtered_data)):
                                                filtered_data[d] = list(filter(None, DDL_solver.filtered_data[d]))

                                            data = classified_data[:] + filtered_data[:]

                                            temp_accu_new_data = {}
                                            for ids in range(len(data)):
                                                temp_accu_new_data[class_order[ids]] = data[ids]


                                        num_nodes.append(N[class_ids])
                                        num_valid_nodes.append(dl.valid_N)
                                        num_literal.append(lit)


                                        clauses.append(len(DDL_solver.cnf.hard))
                                        soft_clauses.append(len(DDL_solver.cnf.soft))
                                        variables.append(DDL_solver.cnf.nv)

                                    else:
                                        if 10 < ubounds[class_ids] - N[class_ids]:
                                            if 10 < 2 * N[class_ids]:
                                                N[class_ids] += 10
                                            else:
                                                N[class_ids] *= 2
                                        else:
                                            N[class_ids] = ubounds[class_ids]

                                        if accu_runtime > 1800:
                                            found = True
                                            solved = False
                                            break


                                else:
                                    # If not a sparse model
                                    found = True
                                    solved = True
                                    sols.append(sol)
                                    num_nodes.append(N[class_ids])
                                    num_valid_nodes.append(dl.valid_N)
                                    num_literal.append(lit)

                                    '''
                                    try:
                                        DDL_solver.cnf.to_file('trees/' + train_filename[:-4] + '/' + str(class_ids) + '.cnf')
                                    except FileNotFoundError as e:
                                        os.makedirs('trees/' + train_filename[:-4])
                                    '''
                                    if maxsat:
                                        clauses.append(len(DDL_solver.cnf.hard))
                                        soft_clauses.append(len(DDL_solver.cnf.soft))
                                    else:
                                        clauses.append(len(DDL_solver.cnf.clauses))
                                    variables.append(DDL_solver.cnf.nv)

                        else:
                            # If not solved, upper bound + 1
                            N[class_ids] += 1

                            if accu_runtime > 1800:
                                found = True
                                solved = False



                    # If the order is based on accuracy
                    # Record the result for each class respectively
                    # Only generate a decision list for the particular class in an order
                    if sep_order in ('accuy', 'cost'):
                        if not solved:
                            Detail = []
                            Detail.append(train_filename)
                            Detail.append(len(data_features))
                            Detail.append(len(data_features[0]))
                            Detail.append(num_class)
                            Detail.append('N')
                            Detail.append('not solved')
                            Detail.append('not solved')
                            Detail.append('not solved')
                            Detail.append('not solved')
                            Detail.append('not solved')
                            Detail.append('not solved')
                            Detail.append('not solved')
                            Detail.append('not solved')
                            Detail.append('not solved')

                            print([Detail])

                        train_sol_rec[curr_pos][class_order[class_ids]] = {'mis_count': nof_miss, 'sol': sol, 'best': False,
                                                        'node': num_nodes[0], 'valid_node': dl.valid_N, 'lit': lit,
                                                        'time': sum(all_runtime[class_ids]), 'data': temp_accu_new_data,
                                                        'cost': nof_used + int(math.ceil(nof_miss / LAM)) if sparse else None}
                        #print('cost', cost)
                        break

                # If the order is not based on accuracy
                if sep_order not in ('accuy', 'cost'):
                    if solved:
                        dl = DL(K, sum(num_nodes))
                        dl.parse_solution(sols=sols,num_nodes=num_valid_nodes)
                        dl.generate_node(num_nodes=num_valid_nodes, class_order=class_order)

                        # Calculate the accuracy for training datasets
                        dl.mis_count = 0
                        dl.validate(data_features, data_classes, num_nodes=num_valid_nodes, num_class=num_class, ml=ml)
                        dl.generate_list(ml)
                        train_accuracy = 1 - dl.mis_count / len(data_classes)

                        # Calculate the accuracy for testing datasets
                        dl.mis_count = 0
                        dl.validate(test_data_features, test_data_classes, num_nodes=num_valid_nodes, num_class=num_class, ml=ml)
                        test_accuracy = 1 - dl.mis_count / len(test_data_classes)

                        #print(dl.mis_count, len(test_data_classes))

                        Detail = []
                        Detail.append(train_filename)
                        Detail.append(len(data_features))
                        Detail.append(len(data_features[0]))
                        Detail.append(num_class)
                        Detail.append('Y')
                        Detail.append(str(dl.valid_N))
                        Detail.append(sum(num_valid_nodes) - sum(num_literal) if sum(num_valid_nodes) != 0 else 1)
                        Detail.append(sum(num_literal))
                        Detail.append(round(train_accuracy * 100, 6))
                        #Detail.append(#round(test_accuracy * 100, 6))
                        Detail.append('%.6f' % accu_runtime)
                        Detail.append(class_order)
                        Detail.append(ml)
                        Detail.append(dl.node)

                        all_order_detail.append(Detail)
                    else:
                        Detail = []
                        Detail.append(train_filename)
                        Detail.append(len(data_features))
                        Detail.append(len(data_features[0]))
                        Detail.append(num_class)
                        Detail.append('N')
                        Detail.append('not solved')
                        Detail.append('not solved')
                        Detail.append('not solved')
                        Detail.append('not solved')
                        Detail.append('not solved')
                        Detail.append('not solved')
                        Detail.append(class_order)
                        Detail.append(ml)
                        Detail.append(dl.node)

                        all_order_detail.append(Detail)


            if sep_order not in ('accuy', 'cost'):
                done = True # Have gone through all orders

            else:

                # If the order is based on accuracy/cost
                if curr_pos == num_class-1: # Have gone through all classes
                    done = True
                    num_nodes = []
                    num_valid_nodes = []
                    lit = 0
                    accu_runtime = 0
                    class_order = final_order[:] + list(set(list(range(1, num_class + 1))) - set(final_order[:]))
                    final_order = class_order[:] # Final order based or accuracy
                    sols = []

                    #
                    for c in range(num_class):
                        for cc in train_sol_rec[c].keys():
                            if c == num_class - 1:
                                train_sol_rec[c][cc]['best'] = True
                            accu_runtime += train_sol_rec[c][cc]['time']
                            if train_sol_rec[c][cc]['best']:
                                num_nodes.append(train_sol_rec[c][cc]['node'])
                                num_valid_nodes.append(train_sol_rec[c][cc]['valid_node'])
                                lit += train_sol_rec[c][cc]['lit']
                                sols.append(train_sol_rec[c][cc]['sol'])

                    #Calculate final accuracy
                    dl = DL(K, sum(num_nodes))
                    dl.parse_solution(sols=sols,num_nodes=num_valid_nodes)
                    dl.generate_node(num_nodes=num_valid_nodes, class_order=final_order)

                    dl.validate(data_features, data_classes, num_nodes=num_valid_nodes, num_class=num_class, ml=ml)
                    dl.generate_list(ml)
                    train_accuracy = 1 - dl.mis_count / len(data_classes)

                    dl.mis_count = 0
                    dl.validate(test_data_features, test_data_classes, num_nodes=num_valid_nodes, num_class=num_class, ml=ml)
                    test_accuracy = 1 - dl.mis_count / len(test_data_classes)
                    #print(dl.mis_count, len(test_data_classes))

                    Detail = []
                    Detail.append(train_filename)
                    Detail.append(len(data_features))
                    Detail.append(len(data_features[0]))
                    Detail.append(num_class)
                    Detail.append('Y')
                    Detail.append(sum(num_valid_nodes))
                    Detail.append(sum(num_valid_nodes) - lit if sum(num_nodes) != 0 else 1) #'rule'
                    Detail.append(lit) # lit
                    Detail.append(round(train_accuracy * 100, 6))
                    #Detail.append(#round(test_accuracy * 100, 6))
                    Detail.append('%.6f' % accu_runtime)
                    Detail.append(final_order)
                    Detail.append(ml)
                    Detail.append(dl.node)

                    all_order_detail.append(Detail)

                else:
                    #If the final decision has not generated


                    # The order is based on accuracy
                    if sep_order == 'accuy':
                        # Record the class with least misclassification for a specific position is a decision list
                        if asc_order == 'desc':
                            min_mis_count = math.inf
                            min_class = None
                            for key in train_sol_rec[curr_pos].keys():
                                if train_sol_rec[curr_pos][key]['mis_count'] <= min_mis_count:
                                    min_mis_count = train_sol_rec[curr_pos][key]['mis_count']
                                    min_class = key

                            train_sol_rec[curr_pos][min_class]['best'] = True
                            best_class = min_class

                        else:
                            assert asc_order == 'asc'
                            # Record the class with most misclassification for a specific position is a decision list
                            max_mis_count = 0
                            max_class = None
                            for key in train_sol_rec[curr_pos].keys():
                                if train_sol_rec[curr_pos][key]['mis_count'] >= max_mis_count:
                                    max_mis_count = train_sol_rec[curr_pos][key]['mis_count']
                                    max_class = key

                            train_sol_rec[curr_pos][max_class]['best'] = True
                            best_class = max_class

                    else:
                        # The order is based on cost
                        assert sep_order == 'cost'
                        if asc_order == 'desc':
                            max_cost = 0
                            max_class = None
                            for key in train_sol_rec[curr_pos].keys():
                                if train_sol_rec[curr_pos][key]['cost'] >= max_cost:
                                    max_cost = train_sol_rec[curr_pos][key]['cost']
                                    max_class = key

                            train_sol_rec[curr_pos][max_class]['best'] = True
                            best_class = max_class
                        else:
                            assert asc_order == 'asc'
                            min_cost = math.inf
                            min_class = None
                            for key in train_sol_rec[curr_pos].keys():
                                if train_sol_rec[curr_pos][key]['cost'] <= min_cost:
                                    min_cost = train_sol_rec[curr_pos][key]['cost']
                                    min_class = key

                            train_sol_rec[curr_pos][min_class]['best'] = True
                            best_class = min_class


                    final_order.append(best_class)  # Order appends one best class
                    tem_order = list(range(1, num_class + 1)) # A possible order for the next step
                    n_order = list(set(tem_order) - set(final_order)) #The order for non-classified classes

                    # Possible orders for the other steps
                    all_class_order = []
                    for c_ids in range(len(n_order)):
                        all_class_order.append(final_order[:] + n_order[c_ids:] + n_order[:c_ids])


                    train_sol_rec[curr_pos+1] = {c_ids: {'mis_count': None, 'sol': None, 'best': False,
                                                    'node': None, 'valid_nodes': None, 'lit': None, 'time': 0,
                                                    'data': None, 'cost': None}
                                        for c_ids in n_order}

                    accu_new_data = train_sol_rec[curr_pos][best_class]['data'] #Used for new step
                    curr_pos += 1

        print(all_order_detail)

    else:
        # For complete models
        #The upper bound
        N = [num_nodes]
        data = (data_features, data_classes) #training data
        sols = [] # Record the solutions
        num_nodes = [] # Number of unused and used nodes
        num_literal = [] # Number of literals excluding leaf nodes(rules)

        #Clauses, variables in CNF
        clauses = []
        variables = []

        #Record the runtime
        all_runtime = []
        accu_runtime = 0

        found = False

        ubounds = [math.ceil((len(data_features[0]) + 1) * len(data_features))]

        while not found:
            #print(N[0], accu_runtime, datetime.now().strftime("%H:%M:%S"))
            DDL_solver = DLSolver(K, N[0], data, LAM=LAM, mis_weight=mis_weight, maxsat=maxsat, sep=sep, ml=ml, sparse=sparse)

            solutions = DDL_solver.solve()
            all_runtime.append(DDL_solver.runtime)
            accu_runtime += DDL_solver.runtime

            if solutions is not None:

                for sol in solutions:

                    sols.append(sol)
                    num_nodes.append(N[0])

                    # Generate a decision list
                    dl = DL(K, N[0], C=num_class)
                    dl.parse_solution(sols=[sol])
                    dl.generate_node()

                    dl.validate(data_features, data_classes, num_nodes=[dl.valid_N])


                    nof_miss, misclas_ids = dl.get_miss(len(data[0]), sol)
                    nof_used = dl.valid_N
                    cost = nof_used + int(math.ceil(nof_miss / LAM))
                    ubounds[0] = min(cost, ubounds[0])
                    #assert nof_miss == dl.mis_count

                    # For size in sparse models
                    if ubounds[0] in (N[0], nof_used):

                        found = True
                        # Generate a decision lsit and calculate accuracy
                        dl.generate_list()
                        lit = dl.get_num_literals()
                        num_literal.append(lit)

                        dl.validate(data_features, data_classes, num_nodes=[dl.valid_N])
                        train_accuracy = 1 - dl.mis_count / len(data[0])

                        # For testing datasets
                        dl.mis_count = 0
                        # print(dl.mis_count)
                        dl.validate(test_data_features, test_data_classes, num_nodes=[dl.valid_N])
                        test_accuracy = 1 - dl.mis_count / len(test_data_features)

                        try:
                            DDL_solver.cnf.to_file('trees/' + train_filename[:-4] + '/whole_model.cnf')
                        except FileNotFoundError as e:
                            os.makedirs('trees/' + train_filename[:-4])

                        if maxsat:
                            clauses.append(len(DDL_solver.cnf.hard))
                        else:
                            clauses.append(len(DDL_solver.cnf.clauses))
                        variables.append(DDL_solver.cnf.nv)

                        Detail = []
                        Detail.append(train_filename)
                        Detail.append(len(data_features))
                        Detail.append(len(data_features[0]))
                        Detail.append(num_class)
                        Detail.append('Y')
                        Detail.append(str(dl.valid_N))
                        Detail.append(dl.valid_N - lit if dl.valid_N != 1 else 1)
                        Detail.append(lit if dl.valid_N != 1 else 0)
                        Detail.append(round(train_accuracy * 100, 6))
                        #Detail.append(#round(test_accuracy * 100, 6))
                        Detail.append('%.6f' % accu_runtime)
                        Detail.append('Memory Peak')
                        Detail.append(ml)
                        Detail.append(dl.node)

                        print([Detail])

                    else:
                        if 10 < ubounds[0] - N[0]:
                            if 10 < 2 * N[0]:
                                N[0] += 10
                            else:
                                N[0] *= 2
                        else:
                            N[0] = ubounds[0]



            else:
                N[0] += 1


