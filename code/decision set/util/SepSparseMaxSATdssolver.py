import numpy as np
from pysat.solvers import Glucose3, Cadical
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
import math

class SepSparseMaxSATdsSolver():
    def __init__(self, K, N, data, lam, mis_weight):
        self.K = K  # The number of features.
        self.N = N  # The number of nodes
        self.selected_features = data[0]

        self.M = len(self.selected_features)
        self.non_selected_features = data[1]
        self.non_M = len(self.non_selected_features)
        self.pi = np.empty(shape=(self.M + self.non_M + 1, self.K + 2), dtype=object)

        self.var_s = np.empty(shape=(self.N + 1, self.K + 2), dtype=object)  # var_S[j,K+1] is leaf
        self.var_v = np.empty(shape=(self.M + self.non_M + 1, self.N + 1),
                              dtype=object)  # var_v[i,j] is one when item i is valid in node j
        self.var_t = np.empty(shape=(self.N + 1), dtype=object)  # var_t[j] The value on node j

        self.var_a0 = np.empty(shape=(self.N + 1, self.K + 2), dtype=object)
        self.var_a1 = np.empty(shape=(self.N + 1, self.K + 2), dtype=object)

        self.var_aux_vl = np.empty(shape=(self.M + self.non_M + 1, self.N + 1), dtype=object)
        self.var_r = np.empty(shape=(self.N + 1, self.K + 1), dtype=object)

        self.var_m = np.empty(shape=(self.M + self.non_M + 1), dtype=object)

        self.LAM = math.ceil(lam * N)
        self.mis_weight = mis_weight

        self.cnf = WCNF()
        self.solver = None
        self.runtime = 0

        self.var2ids = {}
        self.ids2var = {}
        self.cnf_ids = 0
        self.all_var_sol = None

        self.basic_keys = ("s", "t", "u")  # DS variables,

        self.var_u = np.empty(shape=(self.N + 1), dtype=object)

        for i in range(1, self.pi.shape[0]):
            for r in range(1, self.pi.shape[1] - 1):
                if i < (self.M + 1):
                    self.pi[i, r] = self.selected_features[i - 1][r - 1]
                else:
                    self.pi[i, r] = self.non_selected_features[i - self.M - 1][r - 1]

        def add_vars_1d(vars, var_name):
            for i in range(0, vars.shape[0]):
                vars[i] = var_name + "[" + str(i) + "]"

        def add_vars_2d(vars, var_name):
            for i in range(0, vars.shape[0]):
                for j in range(0, vars.shape[1]):
                    vars[i, j] = var_name + "[" + str(i) + "," + str(j) + "]"

        add_vars_2d(self.var_v, "v")
        add_vars_2d(self.var_s, "s")
        add_vars_1d(self.var_t, "t")
        add_vars_2d(self.var_a0, "a0")
        add_vars_2d(self.var_a1, "a1")
        add_vars_2d(self.var_aux_vl, "aux_vl")
        add_vars_2d(self.var_r, "r")

        add_vars_1d(self.var_u, "u")
        add_vars_1d(self.var_m, "m")


    def get_cnf_id(self, var, value=1):
        if var not in self.var2ids:
            self.cnf_ids += 1
            self.var2ids[var] = self.cnf_ids

            assert self.cnf_ids not in self.ids2var
            self.ids2var[self.cnf_ids] = var

        return self.var2ids[var] if value == 1 else -self.var2ids[var]

    def encode_constraints(self):  # Encode the constraints in paper
        def cons_1(self):  # Exactly one S[j1] + S[j2]... + S[jf] + S[jc]
            for j in range(1, self.N + 1):
                or_f = []
                for f in range(1, self.K + 2):
                    or_f.append(self.get_cnf_id(self.var_s[j, f]))
                self.cnf.append(or_f )

                for a in range(len(or_f)):
                    for b in range(a + 1, len(or_f)):
                        new_f = []
                        new_f.append(-or_f[a])
                        new_f.append(-or_f[b])
                        self.cnf.append(new_f)

        def cons_2(self):
            for j in range(1, self.N + 1):
                for r in range(1, self.K + 2):
                    # a
                    f1 = []
                    f1.append(self.get_cnf_id(self.var_a0[j, r], '-'))
                    f1.append(self.get_cnf_id(self.var_s[j, r]))
                    self.cnf.append(f1)

                    f2 = []
                    f2.append(self.get_cnf_id(self.var_a0[j, r], '-'))
                    f2.append(self.get_cnf_id(self.var_t[j], '-'))
                    self.cnf.append(f2)

                    f3 = []
                    f3.append(self.get_cnf_id(self.var_a0[j, r]))
                    f3.append(self.get_cnf_id(self.var_s[j, r], '-'))
                    f3.append(self.get_cnf_id(self.var_t[j]))
                    self.cnf.append(f3)

                    # b
                    f1 = []
                    f1.append(self.get_cnf_id(self.var_a1[j, r], '-'))
                    f1.append(self.get_cnf_id(self.var_s[j, r]))
                    self.cnf.append(f1)

                    f2 = []
                    f2.append(self.get_cnf_id(self.var_a1[j, r], '-'))
                    f2.append(self.get_cnf_id(self.var_t[j]))
                    self.cnf.append(f2)

                    f3 = []
                    f3.append(self.get_cnf_id(self.var_a1[j, r]))
                    f3.append(self.get_cnf_id(self.var_s[j, r], '-'))
                    f3.append(self.get_cnf_id(self.var_t[j], '-'))
                    self.cnf.append(f3)


        def cons_3(self): # s[self.N, k+1]
            self.cnf.append([self.get_cnf_id(self.var_s[self.N, self.K+1])])


        def cons_4(self):  # All data are valid at the first node
            for i in range(1, self.M + self.non_M + 1):
                cnf_id = [self.get_cnf_id(self.var_v[i, 1])]
                self.cnf.append(cnf_id)

        def cons_5(self):
            for i in range(1, self.M + self.non_M + 1):
                for j in range(1, self.N):
                    f1 = []
                    f1.append(self.get_cnf_id(self.var_v[i, j+1], '-'))
                    f1.append(self.get_cnf_id(self.var_s[j, self.K+1]))
                    f1.append(self.get_cnf_id(self.var_v[i, j]))
                    self.cnf.append(f1)

                    f2 = []
                    f2.append(self.get_cnf_id(self.var_v[i, j + 1], '-'))
                    f2.append(self.get_cnf_id(self.var_s[j, self.K + 1]))
                    for f in range(1, self.K+1):
                        if self.pi[i, f] == 0:
                            f2.append(self.get_cnf_id(self.var_a0[j, f]))
                        else:
                            assert self.pi[i, f] == 1
                            f2.append(self.get_cnf_id(self.var_a1[j, f]))
                    self.cnf.append(f2)

                    f3 = []
                    f3.append(self.get_cnf_id(self.var_v[i, j + 1]))
                    f3.append(self.get_cnf_id(self.var_s[j, self.K + 1], '-'))
                    self.cnf.append(f3)

                    f4 = []
                    f4.append(self.get_cnf_id(self.var_v[i, j + 1]))
                    f4.append(self.get_cnf_id(self.var_v[i, j], '-'))
                    for r in range(1, self.K+1):
                        ff = f4.copy()
                        if self.pi[i, r] == 0:
                            ff.append(self.get_cnf_id(self.var_a0[j, r], '-'))
                        else:
                            assert self.pi[i, r] == 1
                            ff.append(self.get_cnf_id(self.var_a1[j, r], '-'))
                        self.cnf.append(ff)

        def cons_6(self): #s[j,c] /\ v[i, j] -> a[jc]
            for i in range(1, self.M + 1):
                for j in range(1, self.N + 1):
                    f1 = []
                    f1.append(self.get_cnf_id(self.var_s[j, self.K + 1], '-'))
                    f1.append(self.get_cnf_id(self.var_v[i, j], '-'))
                    if self.pi[i, self.K + 1] == 0:
                        f1.append(self.get_cnf_id(self.var_a0[j, self.K + 1]))
                    else:
                        assert self.pi[i, self.K + 1] == 1
                        f1.append(self.get_cnf_id(self.var_a1[j, self.K + 1]))
                    self.cnf.append(f1)

        def cons_7(self): #Exist s[jc] /\ v[ij]
            #Auxiliary variables vl[i,j] <-> s[jc] /\ v[ij]
            for i in range(1, self.M + self.non_M + 1):
                for j in range(1, self.N + 1):
                    f1 = []
                    f1.append(self.get_cnf_id(self.var_aux_vl[i, j], '-'))
                    f1.append(self.get_cnf_id(self.var_s[j, self.K + 1]))
                    self.cnf.append(f1)

                    f2 = []
                    f2.append(self.get_cnf_id(self.var_aux_vl[i, j], '-'))
                    f2.append(self.get_cnf_id(self.var_v[i, j]))
                    self.cnf.append(f2)

                    f3 = []
                    f3.append(self.get_cnf_id(self.var_aux_vl[i, j]))
                    f3.append(self.get_cnf_id(self.var_s[j, self.K + 1], '-'))
                    f3.append(self.get_cnf_id(self.var_v[i, j], '-'))
                    self.cnf.append(f3)

            # Exist vl[i,j]
            for i in range(1, self.M + 1):
                or_f = []
                for j in range(1, self.N + 1):
                    or_f.append(self.get_cnf_id(self.var_aux_vl[i, j]))
                self.cnf.append(or_f)

            #For split model
            def cons_7s(self):
                for ii in range(self.M + 1, self.M + self.non_M + 1 ):
                    for j in range(1, self.N + 1):
                        self.cnf.append([self.get_cnf_id(self.var_aux_vl[ii, j], '-')])
            cons_7s(self)

        def cons_8(self):  # S[j+1,f] -> l[j] \/ exists S[j,f']
            for j in range(1, self.N):
                for r in range(1, self.K + 1):
                    or_f = []
                    or_f.append(self.get_cnf_id(self.var_s[j, self.K+1]))
                    or_f.append(self.get_cnf_id(self.var_s[j, r], '-'))
                    for q in range(r+1, self.K+2):
                        or_f.append(self.get_cnf_id(self.var_s[j + 1, q]))

                    self.cnf.append(or_f)

        def cons_9(self):
            for r in range(1, self.K+2):
                for j1 in range(1, self.N+1):
                    for j2 in range(j1 + 1, self.N+1):
                        or_f = []
                        or_f.append(self.get_cnf_id(self.var_s[j1 - 1, self.K + 1], '-'))
                        or_f.append(self.get_cnf_id(self.var_s[j2 - 1, self.K + 1], '-'))
                        or_f.append(self.get_cnf_id(self.var_s[j1, r], '-'))

                        for q in range(r, self.K+2):
                            or_f.append(self.get_cnf_id(self.var_s[j2, q]))

                        self.cnf.append(or_f)

        def cons_9_Max(self):
            for r in range(1, self.K+2):
                for j1 in range(1, self.N+1):
                    for j2 in range(j1 + 1, self.N+1):
                        or_f = []
                        or_f.append(self.get_cnf_id(self.var_s[j1 - 1, self.K + 1], '-'))
                        or_f.append(self.get_cnf_id(self.var_s[j2 - 1, self.K + 1], '-'))
                        or_f.append(self.get_cnf_id(self.var_s[j1, r], '-'))

                        for q in range(r, self.K+2):
                            or_f.append(self.get_cnf_id(self.var_s[j2, q]))

                        or_f.append(self.get_cnf_id(self.var_u[j2]))

                        self.cnf.append(or_f)

        def cons_10(self):
            self.cnf.append([self.get_cnf_id(self.var_s[0, self.K+1])])

        def cons_11(self):  # Exactly one u[j] + S[j1] + S[j2]... + S[jf] + S[jc]
            for j in range(1, self.N + 1):
                or_f = []
                or_f.append(self.get_cnf_id(self.var_u[j]))
                for r in range(1, self.K + 2):
                    or_f.append(self.get_cnf_id(self.var_s[j, r]))
                self.cnf.append(or_f)

                for a in range(len(or_f)):
                    for b in range(a + 1, len(or_f)):
                        new_f = []
                        new_f.append(-or_f[a])
                        new_f.append(-or_f[b])
                        self.cnf.append(new_f)

        def cons_12(self):  # Exactly one u[j] + S[j1] + S[j2]... + S[jf] + S[jc]
            for j in range(1, self.N):
                f1 = []
                f1.append(self.get_cnf_id(self.var_u[j], '-'))
                f1.append(self.get_cnf_id(self.var_u[j+1]))
                self.cnf.append(f1)

        def cons_12(self):  # u[j] -> u[j+1]
            for j in range(1, self.N):
                f1 = []
                f1.append(self.get_cnf_id(self.var_u[j], '-'))
                f1.append(self.get_cnf_id(self.var_u[j+1]))
                self.cnf.append(f1)

        def cons_13(self): # u[j+1] -> u[j] \/ s[j, c]
            for j in range(1, self.N):
                f1 = []
                f1.append(self.get_cnf_id(self.var_u[j + 1], '-'))
                f1.append(self.get_cnf_id(self.var_u[j]))
                f1.append(self.get_cnf_id(self.var_s[j, self.K + 1]))
                self.cnf.append(f1)

        def cons_14(self): # u[j+1] -> u[j] \/ s[j, c]
            for j in range(1, self.N):
                f1 = []
                f1.append(self.get_cnf_id(self.var_u[j + 1], '-'))
                f1.append(self.get_cnf_id(self.var_u[self.N]))
                f1.append(self.get_cnf_id(self.var_s[self.N, self.K + 1]))
                self.cnf.append(f1)

        def cons_15(self): #s[j,c] /\ v[i, j] -> a[jc] \/ m[i]
            for i in range(1, self.M + 1):
                for j in range(1, self.N + 1):
                    f1 = []
                    f1.append(self.get_cnf_id(self.var_s[j, self.K + 1], '-'))
                    f1.append(self.get_cnf_id(self.var_v[i, j], '-'))
                    if self.pi[i, self.K + 1] == 0:
                        f1.append(self.get_cnf_id(self.var_a0[j, self.K + 1]))
                    else:
                        assert self.pi[i, self.K + 1] == 1
                        f1.append(self.get_cnf_id(self.var_a1[j, self.K + 1]))

                    f1.append(self.get_cnf_id(self.var_m[i]))
                    self.cnf.append(f1)


        def cons_16(self): #Exist s[jc] /\ v[ij]
            #Auxiliary variables vl[i,j] <-> s[jc] /\ v[ij]
            for i in range(1, self.M + 1):
                for j in range(1, self.N + 1):
                    f1 = []
                    f1.append(self.get_cnf_id(self.var_aux_vl[i, j], '-'))
                    f1.append(self.get_cnf_id(self.var_s[j, self.K + 1]))
                    self.cnf.append(f1)

                    f2 = []
                    f2.append(self.get_cnf_id(self.var_aux_vl[i, j], '-'))
                    f2.append(self.get_cnf_id(self.var_v[i, j]))
                    self.cnf.append(f2)

                    f3 = []
                    f3.append(self.get_cnf_id(self.var_aux_vl[i, j]))
                    f3.append(self.get_cnf_id(self.var_s[j, self.K + 1], '-'))
                    f3.append(self.get_cnf_id(self.var_v[i, j], '-'))
                    self.cnf.append(f3)

            # Exist mi \/ Any vl[i,j]
            for i in range(1, self.M + 1):
                or_f = []
                or_f.append(self.get_cnf_id(self.var_m[i]))
                for j in range(1, self.N + 1):
                    or_f.append(self.get_cnf_id(self.var_aux_vl[i, j]))
                self.cnf.append(or_f)

            # First node is not leaf
            self.cnf.append([self.get_cnf_id(self.var_s[1, self.K+1], '-')])

            #s[j, c] -> not s[j+1,c]
            '''
            for j in range(1, self.N):
                or_f = []
                or_f.append(self.get_cnf_id(self.var_s[j, self.K+1], '-'))
                or_f.append(self.get_cnf_id(self.var_s[j + 1, self.K + 1], '-'))
                self.cnf.append(or_f)
            '''


        def cons_17(self):
            cons_16(self) #Exist s[jc] /\ v[ij]

            # Any s[jc] /\ v[ij] -> m[i]
            '''
            for i in range(self.M + 1, self.M + self.non_M + 1):
                or_f = []
                or_f.append(self.get_cnf_id(self.var_m[i]))
                for j in range(1, self.N + 1):
                    ff = or_f.copy()
                    ff.append(self.get_cnf_id(self.var_aux_vl[i, j], '-'))
                    self.cnf.append(ff)
            '''

            for i in range(self.M + 1, self.M + self.non_M + 1):
                or_f = []
                or_f.append(self.get_cnf_id(self.var_m[i]))
                for j in range(1, self.N + 1):
                    ff = or_f.copy()
                    ff.append(self.get_cnf_id(self.var_s[j, self.K + 1], '-'))
                    ff.append(self.get_cnf_id(self.var_v[i, j], '-'))
                    self.cnf.append(ff)

        cons_2(self)
        cons_4(self)
        cons_5(self)

        def symmetry_breaking_for_MaxSAT(self):
            cons_8(self)
            cons_9_Max(self)
            cons_10(self)

        def Sep_Sparse_MaxSAT_model(self):
            cons_11(self)
            cons_12(self)
            cons_13(self)
            cons_14(self)
            cons_17(self)

            for i in range(1, self.M + self.non_M + 1):
                self.cnf.append([self.get_cnf_id(self.var_m[i], '-')], weight=self.mis_weight)

            for j in range(1, self.N + 1):
                self.cnf.append([self.get_cnf_id(self.var_u[j])], weight=self.LAM)

        #symmetry_breaking_for_MaxSAT(self)
        Sep_Sparse_MaxSAT_model(self)

    def solve(self):
        self.encode_constraints()
        self.all_var_sol = []
        ds_var_sol = []
        self.solver = RC2(self.cnf)

        s = self.solver.compute()

        self.runtime = self.solver.oracle_time()

        if s is None:
            return None
        else:
            misclas =[]
            for ids in s:
                var_name = self.ids2var[abs(ids)]
                var_val = 1 if ids > 0 else 0
                self.all_var_sol.append((var_name, var_val))
                if var_name[0] in self.basic_keys:
                    ds_var_sol.append((var_name, var_val))

                if var_name[0].strip().lower() == 'm':
                    misclas.append((var_name, var_val))
            print(misclas)

            return [ds_var_sol]








