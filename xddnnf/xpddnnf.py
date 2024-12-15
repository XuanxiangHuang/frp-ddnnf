#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   d-DNNF Classifiers explainer
#   Author: Xuanxiang Huang
#
################################################################################
import numpy as np
import pandas as pd
import resource
import networkx as nx
from pysat.formula import IDPool
from pysat.solvers import Solver
from xddnnf.parsing import *
################################################################################


class XpdDnnf(object):
    """
        Explain d-DNNF classifier.
    """
    def __init__(self, nn, ddnnf, root, leafs, l2l, verb=0):
        self.nn = nn            # num of nodes
        self.ddnnf = ddnnf      # ddnnf graph
        self.root = root        # root node
        self.leafs = leafs      # leaf nodes
        self.lit2leaf = l2l     # map a literal to its corresponding leaf node
        self.nf = None          # num of features
        self.feats = None       # features
        self.domtype = None     # type of domain ('discrete', 'continuous')
        self.bfeats = None      # binarized features (grouped together)
        self.bflits = None      # literal of binarized feature (grouped together)
        self.lits = None        # literal converted from instance
        self.verbose = verb     # verbose level

    @classmethod
    def from_file(cls, filename, verb=0):
        """
            Load (smooth) d-DNNF model from .ddnnf format file.
            Given .ddnnf file MUST CONTAIN a smooth d-DNNF model.

            :param filename: file in .ddnnf format.
            :param verb: verbose level
            :return: (smooth) d-DNNF model.
        """

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        # filtering out comment lines (those that start with '#')
        lines = list(filter(lambda l: (not (l.startswith('#') or l.strip() == '')), lines))

        lit2leaf = dict()
        leaf = []
        lits = []
        t_nds = []
        nt_nds = []
        edges = []
        index = 0

        assert (lines[index].strip().startswith('NN:'))
        n_nds = (lines[index].strip().split())[1]
        index += 1

        assert (lines[index].strip().startswith('NV:'))
        n_vars = (lines[index].strip().split())[1]
        index += 1

        assert (lines[index].strip().startswith('Root:'))
        root = (lines[index].strip().split())[1]
        index += 1

        assert (lines[index].strip().startswith('TDef:'))
        index += 1

        while not lines[index].strip().startswith('NTDef:'):
            nd, t = lines[index].strip().split()
            leaf.append(int(nd))
            if t == 'F' or t == 'T':
                t_nds.append(tuple((int(nd), {'label': t})))
                lit2leaf.update({t: int(nd)})
            else:
                t_nds.append(tuple((int(nd), {'label': int(t)})))
                lit2leaf.update({int(t): int(nd)})
                if abs(int(t)) not in lits:
                    lits.append(abs(int(t)))
            index += 1

        assert (lines[index].strip().startswith('NTDef:'))
        index += 1

        while index < len(lines):
            string = lines[index].strip().split()
            nd = string[0]
            n_type = string[1]
            assert n_type in ('OR', 'AND')
            nt_nds.append(tuple((int(nd), {'label': n_type})))
            chds = string[3:]
            assert len(chds) == int(string[2])
            for chd in chds:
                edges.append(tuple((int(nd), int(chd))))
            index += 1

        assert (len(t_nds) + len(nt_nds)) == int(n_nds)
        assert len(lits) == int(n_vars)

        G = nx.DiGraph()
        G.add_nodes_from(t_nds)
        G.add_nodes_from(nt_nds)
        G.add_edges_from(edges)

        return cls(int(n_nds), G, int(root), leaf, lit2leaf, verb)

    def parse_feature_map(self, map_file):
        """
            Invoking parsing.parse_feature_map to parse map_file.
        """
        nf, feats, domtype, bfeats, bflits = parse_feature_map(map_file)

        self.nf = nf
        self.feats = feats
        self.domtype = domtype
        self.bfeats = bfeats
        self.bflits = bflits

        if self.verbose == 2:
            print(f"##### parse feature map #####")
            for f, v, l, dtype in zip(feats, bfeats, bflits, domtype):
                print(f"feat: {f}, val: {v}, lit: {l}, type: {dtype}")

    def parse_instance(self, inst):
        """
            Invoking parsing.parse_instance to parse instance file.
        """
        lits = parse_instance(self.nf, self.domtype, self.bfeats, self.bflits, inst)

        self.lits = lits

    def dfs_postorder(self, root):
        """
            Iterate through nodes in depth first search (DFS) post-order.

            :param root: a node of d-DNNF.
            :return: a set of nodes in DFS-post-order.
        """

        #####################################################
        def _dfs_postorder(ddnnf, nd, visited):
            if ddnnf.out_degree(nd):
                for chd in ddnnf.successors(nd):
                    yield from _dfs_postorder(ddnnf, chd, visited)
            if nd not in visited:
                visited.add(nd)
                yield nd
        #####################################################
        yield from _dfs_postorder(self.ddnnf, root, set())

    def get_prediction(self):
        """
            Return prediction of lits (which corresponds to the given instance).

            :return:
        """
        ddnnf = self.ddnnf
        assign = dict()
        for leaf in self.leafs:
            if ddnnf.nodes[leaf]['label'] == 'F':
                assign.update({leaf: 0})
            elif ddnnf.nodes[leaf]['label'] == 'T':
                assign.update({leaf: 1})
        for lit in self.lits:
            if lit:
                for ele in lit:
                    if ele in self.lit2leaf:
                        assign.update({self.lit2leaf[ele]: 1})
                    if -ele in self.lit2leaf:
                        assign.update({self.lit2leaf[-ele]: 0})

        assert len(assign) == len(self.leafs)

        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND'\
                    or ddnnf.nodes[nd]['label'] == 'OR':
                tmp = [assign[chd] for chd in ddnnf.successors(nd)]
                if ddnnf.nodes[nd]['label'] == 'AND':
                    if 0 in tmp:
                        assign.update({nd: 0})
                    else:
                        assign.update({nd: 1})
                else:
                    if 1 in tmp:
                        assign.update({nd: 1})
                    else:
                        assign.update({nd: 0})

        assert assign[self.root] == 1 or assign[self.root] == 0
        return assign[self.root]

    def check_ICoVa(self, univ, va=True):
        """
            Given a list of universal features, check inconsistency or validity.

            :param univ: a list of universal features.
            :param va: True if check validity else check inconsistency.
            :return: True if pass the check
        """
        ddnnf = self.ddnnf
        assign = dict()
        n_univ_var = 0

        for leaf in self.leafs:
            if ddnnf.nodes[leaf]['label'] == 'F':
                assign.update({leaf: 0})
            elif ddnnf.nodes[leaf]['label'] == 'T':
                assign.update({leaf: 1})

        for i in range(self.nf):
            lit = self.lits[i]
            if univ[i]:
                for ele in lit:
                    if ele in self.lit2leaf or -ele in self.lit2leaf:
                        n_univ_var += 1
            else:
                for ele in lit:
                    if ele in self.lit2leaf:
                        assign.update({self.lit2leaf[ele]: 1})
                    if -ele in self.lit2leaf:
                        assign.update({self.lit2leaf[-ele]: 0})

        for leaf in self.leafs:
            if leaf not in assign:
                assign.update({leaf: 1})

        assert len(assign) == len(self.leafs)

        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND' \
                    or ddnnf.nodes[nd]['label'] == 'OR':
                if ddnnf.nodes[nd]['label'] == 'AND':
                    num = 1
                    for chd in ddnnf.successors(nd):
                        num *= assign[chd]
                    assign.update({nd: num})
                else:
                    num = 0
                    for chd in ddnnf.successors(nd):
                        num += assign[chd]
                    assign.update({nd: num})

        n_model = assign[self.root]
        assert n_univ_var >= 0

        if va:
            return n_model == 2 ** n_univ_var
        else:
            return n_model == 0

    def reachable(self, univ, pred):
        """
            Check if desired prediction/class is reachable.

            :param univ: list of universal features.
            :param pred: desired prediction.
            :return: True if reachable else False.
        """
        ddnnf = self.ddnnf
        assign = dict()
        n_univ_var = 0

        for leaf in self.leafs:
            if ddnnf.nodes[leaf]['label'] == 'F':
                assign.update({leaf: 0})
            elif ddnnf.nodes[leaf]['label'] == 'T':
                assign.update({leaf: 1})

        for i in range(self.nf):
            lit = self.lits[i]
            if univ[i]:
                for ele in lit:
                    if ele in self.lit2leaf or -ele in self.lit2leaf:
                        n_univ_var += 1
            else:
                for ele in lit:
                    if ele in self.lit2leaf:
                        assign.update({self.lit2leaf[ele]: 1})
                    if -ele in self.lit2leaf:
                        assign.update({self.lit2leaf[-ele]: 0})

        for leaf in self.leafs:
            if leaf not in assign:
                assign.update({leaf: 1})

        assert len(assign) == len(self.leafs)

        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND' \
                    or ddnnf.nodes[nd]['label'] == 'OR':
                if ddnnf.nodes[nd]['label'] == 'AND':
                    num = 1
                    for chd in ddnnf.successors(nd):
                        num *= assign[chd]
                    assign.update({nd: num})
                else:
                    num = 0
                    for chd in ddnnf.successors(nd):
                        num += assign[chd]
                    assign.update({nd: num})

        n_model = assign[self.root]
        assert n_univ_var >= 0

        if pred:
            return n_model != 0
        else:
            return n_model != 2 ** n_univ_var

    def find_axp(self, fixed=None):
        """
            Compute one abductive explanation (Axp).

            :param fixed: a list of features declared as fixed.
            :return: one abductive explanation,
                        each element in the return Axp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        pred = self.get_prediction()
        # get/create fix array
        if not fixed:
            fix = [True] * self.nf
        else:
            fix = fixed.copy()
        assert (len(fix) == self.nf)

        for i in range(self.nf):
            if fix[i]:
                fix[i] = not fix[i]
                if (pred and not self.check_ICoVa([not v for v in fix], va=True)) or \
                        (not pred and not self.check_ICoVa([not v for v in fix], va=False)):
                    fix[i] = not fix[i]

        axp = [i for i in range(self.nf) if fix[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp: {axp}")
            elif self.verbose == 2:
                print(f"Axp: {axp} ({[self.feats[i] for i in axp]})")
            print("Runtime: {0:.3f}".format(time))

        return axp

    def find_cxp(self, universal=None):
        """
            Compute one contrastive explanation (Cxp).

            :param universal: a list of features declared as universal.
            :return: one contrastive explanation,
                        each element in the return Cxp is a feature index.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        pred = self.get_prediction()
        # get/create univ array
        if not universal:
            univ = [True] * self.nf
        else:
            univ = universal.copy()
        assert (len(univ) == self.nf)

        for i in range(self.nf):
            if univ[i]:
                univ[i] = not univ[i]
                if (pred and self.check_ICoVa(univ, va=True)) or \
                        (not pred and self.check_ICoVa(univ, va=False)):
                    univ[i] = not univ[i]

        cxp = [i for i in range(self.nf) if univ[i]]

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose:
            if self.verbose == 1:
                print(f"Cxp: {cxp}")
            elif self.verbose == 2:
                print(f"Cxp: {cxp} ({[self.feats[i] for i in cxp]})")
            print("Runtime: {0:.3f}".format(time))

        return cxp

    def enum_exps(self):
        """
            Enumerate all (abductive and contrastive) explanations, using MARCO algorithm.

            :return: a list of all Axps, a list of all Cxps.
        """

        #########################################
        vpool = IDPool()

        def new_var(name):
            """
                Inner function,
                Find or new a PySAT variable.
                See PySat.

                :param name: name of variable
                :return: index of variable
            """
            return vpool.id(f'{name}')

        #########################################

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        pred = self.get_prediction()

        num_axps = 0
        num_cxps = 0
        axps = []
        cxps = []

        for i in range(self.nf):
            new_var(f'u_{i}')
        # initially all features are fixed
        univ = [False] * self.nf

        with Solver(name="glucose4") as slv:
            while slv.solve():
                # first model is empty
                model = slv.get_model()
                for lit in model:
                    name = vpool.obj(abs(lit)).split(sep='_')
                    univ[int(name[1])] = False if lit < 0 else True
                if self.reachable(univ, not pred):
                    cxp = self.find_cxp(univ)
                    slv.add_clause([-new_var(f'u_{i}') for i in cxp])
                    num_cxps += 1
                    cxps.append(cxp)
                else:
                    fix = [not i for i in univ]
                    axp = self.find_axp(fix)
                    slv.add_clause([new_var(f'u_{i}') for i in axp])
                    num_axps += 1
                    axps.append(axp)

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if self.verbose == 1:
            print('#AXp:', num_axps)
            print('#CXp:', num_cxps)
            print("Runtime: {0:.3f}".format(time))

        return axps, cxps

    def check_one_axp(self, axp):
        """
            Check if given axp is 1) a weak AXp and 2) subset-minimal.

            :param axp: given axp.
            :return: true if given axp is an AXp
                        else false.
        """
        pred = self.get_prediction()
        fix = [False] * self.nf
        for i in axp:
            fix[i] = True
        # 1) axp is a weak AXp
        if (pred and not self.check_ICoVa([not v for v in fix], va=True)) or \
                (not pred and not self.check_ICoVa([not v for v in fix], va=False)):
            print(f'given axp {axp} is not a weak AXp')
            return False
        # 2) axp is subset-minimal
        for i in range(self.nf):
            if fix[i]:
                fix[i] = not fix[i]
                if (pred and not self.check_ICoVa([not v for v in fix], va=True)) or \
                        (not pred and not self.check_ICoVa([not v for v in fix], va=False)):
                    fix[i] = not fix[i]
                else:
                    print(f'given axp {axp} is not subset-minimal')
                    return False
        return True

    def check_one_cxp(self, cxp):
        """
            Check if given cxp is 1) a weak CXp and 2) subset-minimal.

            :param cxp: given cxp.
            :return: true if given cxp is an CXp
                        else false.
        """
        pred = self.get_prediction()
        univ = [False] * self.nf
        for i in cxp:
            univ[i] = True
        # 1) cxp is a weak CXp
        if (pred and self.check_ICoVa(univ, va=True)) or \
                (not pred and self.check_ICoVa(univ, va=False)):
            print(f'given cxp {cxp} is not a weak CXp')
            return False
        # 2) cxp is subset-minimal
        for i in range(self.nf):
            if univ[i]:
                univ[i] = not univ[i]
                if (pred and self.check_ICoVa(univ, va=True)) or \
                        (not pred and self.check_ICoVa(univ, va=False)):
                    univ[i] = not univ[i]
                else:
                    print(f'given cxp {cxp} is not subset-minimal')
                    return False
        return True

    def is_decomposable(self):
        """
            Check if d-DNNF is decomposable

            :return: True if decomposable
        """
        ddnnf = self.ddnnf
        scope = dict()
        for leaf in self.leafs:
            if ddnnf.nodes[leaf]['label'] == 'F':
                scope.update({leaf: frozenset()})
            elif ddnnf.nodes[leaf]['label'] == 'T':
                scope.update({leaf: frozenset()})
            else:
                lit = ddnnf.nodes[leaf]['label']
                scope.update({leaf: frozenset({abs(lit)})})

        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND' \
                    or ddnnf.nodes[nd]['label'] == 'OR':
                chd_var = [scope.get(chd) for chd in ddnnf.successors(nd)]
                if ddnnf.nodes[nd]['label'] == 'AND':
                    for i in range(len(chd_var)):
                        for j in range(i + 1, len(chd_var)):
                            if not chd_var[i].isdisjoint(chd_var[j]):
                                return False
                tmp = frozenset()
                for ele in chd_var:
                    tmp = tmp.union(ele)
                scope.update({nd: tmp})
        return True

    def is_smooth(self):
        """
            Check if d-DNNF is smooth

            :return: True if smooth
        """
        ddnnf = self.ddnnf
        scope = dict()
        for leaf in self.leafs:
            if ddnnf.nodes[leaf]['label'] == 'F':
                scope.update({leaf: frozenset()})
            elif ddnnf.nodes[leaf]['label'] == 'T':
                scope.update({leaf: frozenset()})
            else:
                lit = ddnnf.nodes[leaf]['label']
                scope.update({leaf: frozenset({abs(lit)})})

        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND' \
                    or ddnnf.nodes[nd]['label'] == 'OR':
                chd_var = [scope.get(chd) for chd in ddnnf.successors(nd)]
                if ddnnf.nodes[nd]['label'] == 'OR':
                    for i in range(len(chd_var)):
                        for j in range(i + 1, len(chd_var)):
                            if chd_var[i] != chd_var[j]:
                                return False
                tmp = frozenset()
                for ele in chd_var:
                    tmp = tmp.union(ele)
                scope.update({nd: tmp})
        return True

    def vars_of_gates(self):
        """
            Collect vars for each gate.

            :return:
        """
        ddnnf = self.ddnnf
        scope = dict()
        for leaf in self.leafs:
            if ddnnf.nodes[leaf]['label'] == 'F':
                scope.update({leaf: frozenset()})
            elif ddnnf.nodes[leaf]['label'] == 'T':
                scope.update({leaf: frozenset()})
            else:
                lit = ddnnf.nodes[leaf]['label']
                scope.update({leaf: frozenset({abs(lit)})})

        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND' \
                    or ddnnf.nodes[nd]['label'] == 'OR':
                chd_var = [scope.get(chd) for chd in ddnnf.successors(nd)]
                tmp = frozenset()
                for ele in chd_var:
                    tmp = tmp.union(ele)
                scope.update({nd: tmp})
        return scope

    def predict(self, instances):
        """
            Return a list of prediction given a list of instances.
            :param instances: a list of (total) instance.
            :return: predictions of these instances
        """
        insts = instances
        if type(instances) == pd.DataFrame:
            insts = instances.to_numpy()
        predictions = []
        for inst in insts:
            self.parse_instance([int(e) for e in list(inst)])
            predictions.append(self.get_prediction())
        return np.array(predictions)

    def predict_prob(self, instances):
        """
            Return a list of probabilities given a list of instances.
            since we only have two classes, we only return the probability of
            class 0 and class 1.
            :param instances: a list of (total) instance.
            :return: predictions of these instances
        """
        insts = instances
        if type(instances) == pd.DataFrame:
            insts = instances.to_numpy()
        predictions = []
        for inst in insts:
            self.parse_instance([int(e) for e in list(inst)])
            if self.get_prediction() == 0:
                # class 0 has probability 1.0, class 1 has probability 0.0
                predictions.append([1.0, 0.0])
            else:
                predictions.append([0.0, 1.0])
        return np.array(predictions)

    def model_counting(self, univ):
        """
            Given a list of universal features, return the number of models.

            :param univ: a list of universal features.
            :return: number of models
        """
        ddnnf = self.ddnnf
        assign = dict()
        n_univ_var = 0

        for leaf in self.leafs:
            if ddnnf.nodes[leaf]['label'] == 'F':
                assign.update({leaf: 0})
            elif ddnnf.nodes[leaf]['label'] == 'T':
                assign.update({leaf: 1})

        for i in range(self.nf):
            lit = self.lits[i]
            if univ[i]:
                for ele in lit:
                    if ele in self.lit2leaf or -ele in self.lit2leaf:
                        n_univ_var += 1
            else:
                for ele in lit:
                    if ele in self.lit2leaf:
                        assign.update({self.lit2leaf[ele]: 1})
                    if -ele in self.lit2leaf:
                        assign.update({self.lit2leaf[-ele]: 0})

        for leaf in self.leafs:
            if leaf not in assign:
                assign.update({leaf: 1})

        assert len(assign) == len(self.leafs)

        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND' \
                    or ddnnf.nodes[nd]['label'] == 'OR':
                if ddnnf.nodes[nd]['label'] == 'AND':
                    num = 1
                    for chd in ddnnf.successors(nd):
                        num *= assign[chd]
                    assign.update({nd: num})
                else:
                    num = 0
                    for chd in ddnnf.successors(nd):
                        num += assign[chd]
                    assign.update({nd: num})

        n_model = assign[self.root]
        assert n_univ_var >= 0
        return n_model
