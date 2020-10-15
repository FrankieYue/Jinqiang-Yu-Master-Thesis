import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

class WholeMaxSATds():
    def __init__(self, K, N):
        self.K = K #The number of features.
        self.N = N #The number of nodes
        self.valid_N = N

        self.s = np.full(shape=(self.N + 1, self.K+2), fill_value=False, dtype=np.bool) # S[i,f]: Node i discriminates on feature f
        self.t = np.full(shape=(self.N + 1), fill_value=False, dtype=np.bool) # t[j] The value on node j
        self.u = np.full(shape=(self.N + 1), fill_value=False, dtype=np.bool)
        self.tree = [[None] for j in range(self.N)]

    def get_num_literals(self):
        num_rule = 0
        for j in range(1, self.N+1):
            if self.s[j, self.K+1]:
                num_rule += 1
        return self.valid_N  - num_rule

    def parse_solution(self, sol):  # solution = [(key, value),(key, value)....] key = (''v[1], 'v[2]', ....'v[N+1]',c'[1]'....) and other variables keys
        var_name_lookup = {"s": self.s, "t": self.t, "u": self.u}
        for var, value in sol:
            var_name = var[: var.find("[")].strip()
            var_index = var[var.find("[") + 1: var.find("]")].strip().split(",")
            assert var_name in var_name_lookup
            if var_name == "t" or var_name == "u":
                var_name_lookup[var_name][int(var_index[0])] = value
            else:

                j = int(var_index[0])
                f = int(var_index[1])
                var_name_lookup[var_name][j, f] = value

    def generate_tree(self):
        for j in range(1, self.N + 1):
            print('u', self.u[j])
            #MAXSAT
            if self.u[j]:
                self.valid_N = j - 1
                self.tree = self.tree[: j-1]
                break
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

    # whole model
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
                        class_ind = 0 if not node_value else 1
                        assert c[class_ind] == 1, 'Wrong solution'
                        
    
    def draw(self, cnt, class_ids, filename):
        nx_tree = nx.Graph()
        nx_tree_label = {}
        pos = [[0, 0]]
        for j in range(1, self.valid_N + 1):
            nx_tree.add_node(j)
            if self.s[j][self.K+1]:
                nx_tree_label[j] = "C" + str(0) if not self.tree[j - 1][1] else "C" + str(1)
            else:
                nx_tree_label[j] = self.tree[j - 1][0] if self.tree[j - 1][1] else "-{0}".format(self.tree[j - 1][0])

            if j + 1 <= self.valid_N:
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