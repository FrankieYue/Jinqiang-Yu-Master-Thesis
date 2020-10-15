import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

class SplitIterativeSATds():
    def __init__(self, K, N):
        self.K = K #The number of features.
        self.N = N #The number of nodes

        self.s = np.full(shape=(self.N+1, self.K+2), fill_value=False, dtype=np.bool) # S[i,f]: Node i discriminates on feature f
        self.t = np.full(shape=(self.N+1), fill_value=False, dtype=np.bool) # t[j] The value on node j
        self.tree = [[None] for j in range(self.N)]

    def get_num_literals(self):
        num_rule = 0
        for j in range(1, self.N+1):
            if self.s[j, self.K+1]:
                num_rule += 1
        return self.N - num_rule

    def parse_solution(self, sol):  # solution = [(key, value),(key, value)....] key = (''v[1], 'v[2]', ....'v[N+1]',c'[1]'....) and other variables keys
        var_name_lookup = {"s": self.s, "t": self.t}
        for var, value in sol:
            var_name = var[: var.find("[")].strip()
            var_index = var[var.find("[") + 1: var.find("]")].strip().split(",")
            assert var_name in var_name_lookup
            if var_name == "l" or var_name == "t":
                var_name_lookup[var_name][int(var_index[0])] = value
            else:

                j = int(var_index[0])
                f = int(var_index[1])
                var_name_lookup[var_name][j, f] = value

    def generate_tree(self):
        for j in range(1, self.N + 1):
            value = self.t[j]
            if self.s[j][self.K+1]:
                feature = "class"
            else:
                feature = None
                for f in range(1, self.K + 1):
                    if self.s[j, f]:
                        feature = "f" + str(f)
                        # print('feature', feature)
                        break
            self.tree[j - 1] = (feature, value)
        print('tree:', self.tree)
    def parse_mul_solution(self, sols, num_nodes): # solution = [(key, value),(key, value)....] key = (''v[1], 'v[2]', ....'v[N+1]',c'[1]'....) and other variables keys
        var_name_lookup = {"s": self.s, "t": self.t}

        sol_ids = 0
        num_pre_nodes = 0
        for sol in sols:
            for var, value in sol:
                var_name = var[: var.find("[")].strip()
                var_index = var[var.find("[")+1 : var.find("]")].strip().split(",")
                assert var_name in var_name_lookup
                if var_name == "t":
                    var_name_lookup[var_name][int(var_index[0]) + num_pre_nodes] = value
                else:
                    assert var_name == "s"
                    j = int(var_index[0])
                    f = int(var_index[1])
                    var_name_lookup[var_name][j + num_pre_nodes, f] = value


            num_pre_nodes += num_nodes[sol_ids]
            sol_ids += 1

    def validate(self, data_features, data_classes):
        for f, c in zip(data_features, data_classes):
            pre_match = True
            pre_leaf = False
            for node in self.tree:
                node_value = node[1]
                if node[0].lower().startswith('f'):
                    node_f = int(node[0][1:])
                    if pre_leaf:
                        pre_leaf = False
                        if f[node_f - 1] == node_value:
                            pre_match = True
                        else:
                            pre_match = False
                    elif pre_match and f[node_f - 1] == node_value:
                        pass
                    else:
                        pre_match = False
                else:
                    assert node[0].lower().strip() == 'class'
                    pre_leaf = True
                    if pre_match:
                        assert c == node_value, 'Wrong solution'

    def com_validate(self, data_features, data_classes, num_nodes, num_class):
        self.N = sum(num_nodes)
        num_pre_nodes = 0
        class_ids = 0
        sol_ids = 0
        for j in range(1, self.N+1):
            next_sol_nodes = num_nodes[sol_ids] if class_ids < num_class else 0
            if j > (num_pre_nodes + next_sol_nodes):
                class_ids += + 1
                num_pre_nodes += next_sol_nodes
                sol_ids += 1
            if self.tree[j - 1][0].lower().strip() == 'class':
                self.tree[j - 1] = ('class' + str(class_ids + 1), True)
        print(self.tree)

        for f, c in zip(data_features, data_classes):
            pre_match = True
            pre_leaf = False
            for node in self.tree:
                node_value = node[1]
                if node[0].lower().startswith('f'):
                    node_f = int(node[0][1:])
                    if pre_leaf:
                        pre_leaf = False
                        if f[node_f - 1] == node_value:
                            pre_match = True
                        else:
                            pre_match = False
                    elif pre_match and f[node_f - 1] == node_value:
                        pass
                    else:
                        pre_match = False
                else:
                    assert node[0][ : 5].lower().strip() == 'class'
                    pre_leaf = True
                    item_class = list(c).index(1)
                    if pre_match:
                       # print((str(item_class + 1), node[0][ 5: ]) )
                        if str(item_class + 1) != node[0][ 5: ]:
                            print('Wrong solution:', str(item_class + 1), node[0][ 5: ])
                       # assert str(item_class + 1) == node[0][ 5: ], 'Wrong solution'

    def draw(self, cnt, class_ids, filename):
        nx_tree = nx.Graph()
        nx_tree_label = {}
        pos = [[0, 0]]
        for j in range(1, self.N + 1):
            nx_tree.add_node(j)
            if self.s[j][self.K + 1]:
                nx_tree_label[j] = "C" + str(class_ids + 1)
            else:
                nx_tree_label[j] = self.tree[j - 1][0] if self.tree[j - 1][1] else "-{0}".format(self.tree[j - 1][0])

            if j + 1 <= self.N:
                nx_tree.add_edge(j, j + 1)

            quotient = (j - 1) // 10
            rem = quotient % 2
            if rem == 0:
                pos.append([j - quotient * 10, - quotient])
            else:
                pos.append([(quotient + 1) * 10 + 1 - j, -quotient])

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        nx.draw(nx_tree, pos, with_labels=True, node_color='blue')
        plt.subplot(2, 1, 2)
        nx.draw(nx_tree, pos, with_labels=True, labels=nx_tree_label, node_color='red')

        try:
            plt.savefig('trees/' + filename + '/new_' + str(cnt) + '.png')
        except FileNotFoundError as e:
            os.makedirs('trees/' + filename)
            plt.savefig('trees/' + filename + '/new_' + str(cnt) + '.png')

        plt.close()

    def com_draw(self, filename, sols, num_nodes, cnt=1):
        self.parse_mul_solution(sols, num_nodes)
        self.generate_tree()

        nx_tree = nx.Graph()
        nx_tree_label = {}
        pos = [[0,0]]

        num_pre_nodes = 0
        class_ids = 0
        sol_ids = 0

        for j in range(1, self.N+1):

            next_sol_nodes = num_nodes[sol_ids] if class_ids < len(sols) else 0
            if j > (num_pre_nodes + next_sol_nodes):
                class_ids += + 1
                num_pre_nodes += next_sol_nodes
                sol_ids += 1

            nx_tree.add_node(j)
            if self.s[j, self.K + 1]:
                nx_tree_label[j] = "C" + str(class_ids + 1)
            else:
                nx_tree_label[j] = self.tree[j - 1][0] if self.tree[j - 1][1] else "-{0}".format(self.tree[j - 1][0])

            if j + 1 <= self.N:
                nx_tree.add_edge(j, j+1)

            quotient = (j-1) // 10
            rem = quotient % 2
            if rem == 0:
                pos.append([j - quotient * 10, - quotient])
            else:
                pos.append([(quotient + 1) * 10 + 1 - j, -quotient])

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        nx.draw(nx_tree, pos, with_labels=True, node_color='blue')
        plt.subplot(2, 1, 2)
        nx.draw(nx_tree, pos, with_labels=True, labels=nx_tree_label, node_color='red')

        try:
            plt.savefig('trees/' + filename + '/new_combined_' + str(cnt) + '.png')
        except FileNotFoundError as e:
            os.makedirs('trees/' + filename)
            plt.savefig('trees/' + filename + '/new_combined_' + str(cnt) + '.png')
        plt.close()
