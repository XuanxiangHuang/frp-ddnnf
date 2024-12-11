#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Feature Membership on d-DNNF Classifiers
#   Author: Xuanxiang Huang
#
################################################################################
import time
import networkx as nx
from pysmt.shortcuts import Equals, Plus, Times, Symbol, LE, And, \
    Int, Solver, Minus, LT, Implies
from pysmt.typing import INT
from xddnnf.parsing import *
from pysmt.exceptions import SolverReturnedUnknownResultError
################################################################################


class FmdDnnf(object):
    """
        Explain d-DNNF classifier.
    """

    def __init__(self, nn, ddnnf, root, leafs, l2l, verb=0):
        self.nn = nn  # num of nodes
        self.ddnnf = ddnnf  # ddnnf graph
        self.root = root  # root node
        self.leafs = leafs  # leaf nodes
        self.lit2leaf = l2l  # map a literal to its corresponding leaf node
        self.nf = None  # num of features
        self.feats = None  # features
        self.domtype = None  # type of domain ('discrete', 'continuous')
        self.bfeats = None  # binarized features (grouped together)
        self.bflits = None  # literal of binarized feature (grouped together)
        self.lits = None  # literal converted from instance
        self.verbose = verb  # verbose level

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
            if ddnnf.nodes[nd]['label'] == 'AND' \
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
        return assign[self.root] == 1

    def two_step(self, pred, feat_id):
        """
            :param pred: prediction.
            :param feat_id: feature id.
            :return: a weak AXp containing desired feature, or None.
        """
        if self.verbose:
            print('(two-step) Feature Membership of d-DNNF into SMT formulas ...')

        ##################### for 0-th replica #####################
        ddnnf = self.ddnnf
        # SMT encoding of d-DNNF
        ###############################################################
        # 1. define m feature selectors (s_i)
        slts = [Symbol(f's_{i}', INT) for i in range(self.nf)]
        cnt = [dict() for _ in range(2)]
        dom = dict()
        for i in range(self.nf):
            appear = 0
            for ele in self.lits[i]:
                if ele in self.lit2leaf or -ele in self.lit2leaf:
                    appear += 1
            dom.update({i: appear})
        ###############################################################
        # 2. define counter for each node
        for k in range(2):
            for nd in self.dfs_postorder(self.root):
                if ddnnf.nodes[nd]['label'] == 'F':
                    cnt[k].update({nd: Int(0)})
                elif ddnnf.nodes[nd]['label'] == 'T':
                    cnt[k].update({nd: Int(1)})
                else:
                    cnt[k].update({nd: Symbol(f'N_{k}_{nd}', INT)})
        ###############################################################
        # 3. 0 <= s_i <= 1
        tmp0 = []
        for s in slts:
            tmp0.append(And(LE(Int(0), s), LE(s, Int(1))))
        for i in range(self.nf):
            if dom[i] == 0:
                tmp0.append(Equals(slts[i], Int(0)))
        fml_slt_domain = And(tmp0)
        ###############################################################
        # 4. constraints for counter of internal nodes
        tmp1 = []
        for k in range(2):
            for nd in self.dfs_postorder(self.root):
                if ddnnf.nodes[nd]['label'] == 'AND' \
                        or ddnnf.nodes[nd]['label'] == 'OR':
                    tmp_chd = []
                    for chd in ddnnf.successors(nd):
                        tmp_chd.append(cnt[k][chd])
                    if ddnnf.nodes[nd]['label'] == 'AND':
                        tmp1.append(Equals(cnt[k][nd], Times(tmp_chd)))
                    else:
                        tmp1.append(Equals(cnt[k][nd], Plus(tmp_chd)))
        fml_internal = And(tmp1)
        ###############################################################
        # 5. constraints for counter of leaf nodes
        tmp2 = []
        for i in range(self.nf):
            for ele in self.lits[i]:
                if ele in self.lit2leaf:
                    leaf = self.lit2leaf[ele]
                    tmp2.append(Equals(cnt[0][leaf], Int(1)))
                if -ele in self.lit2leaf:
                    nleaf = self.lit2leaf[-ele]
                    tmp2.append(Equals(cnt[0][nleaf], Minus(Int(1), slts[i])))
        for i in range(self.nf):
            for ele in self.lits[i]:
                if ele in self.lit2leaf:
                    leaf = self.lit2leaf[ele]
                    tmp2.append(Equals(cnt[1][leaf], Int(1)))
                if -ele in self.lit2leaf:
                    nleaf = self.lit2leaf[-ele]
                    if i == feat_id:
                        tmp2.append(Equals(cnt[1][nleaf], Int(1)))
                    else:
                        tmp2.append(Equals(cnt[1][nleaf], Minus(Int(1), slts[i])))
        fml_leaf = And(tmp2)
        ###############################################################
        # 6. counter of root node
        tmp3 = []
        if pred:
            tmp3.append(Equals(cnt[0][self.root],
                               Times(Minus(Int(2), slts[i]) for i in range(self.nf) if dom[i])))
        else:
            tmp3.append(Equals(cnt[0][self.root], Int(0)))
        fml_out = And(tmp3)
        tmp4 = []
        if pred:
            tmp4.append(LT(cnt[1][self.root], Times(Int(2), cnt[0][self.root])))
        else:
            tmp4.append(LT(Int(0), cnt[1][self.root]))
        fml_check_axp = And(tmp4)
        ###############################################################
        # 7. add all constraints
        fml = And(fml_slt_domain, fml_leaf, fml_internal,
                  fml_out, Equals(slts[feat_id], Int(1)), fml_check_axp)
        weakaxp = []
        failed = False
        num_atoms = len(slts) + len(cnt[0]) + len(cnt[1])
        num_fmls = len(tmp0) + len(tmp1) + len(tmp2) + len(tmp3) + len(tmp4) + 1
        ###############################################################
        if self.verbose:
            print('Start solving ...')
        time_solving_start = time.process_time()
        try:
            with Solver(name="z3", solver_options={'timeout': 300000}) as solver:
                solver.add_assertion(fml)
                if solver.solve():
                    val = [solver.get_value(s) for s in slts]
                    weakaxp = [i for i in range(self.nf) if val[i] == Int(1)]
                    assert feat_id in weakaxp
                    weakaxp.sort()
                    if self.verbose >= 1:
                        print(f"weak AXp: {weakaxp}")
        except SolverReturnedUnknownResultError:
            print('TIMEOUT(300 secs)')
            failed = True
        time_solving_end = time.process_time() - time_solving_start
        if self.verbose:
            print(f"Solving time: {time_solving_end:.1f} secs")

        return weakaxp, num_atoms, num_fmls, failed, time_solving_end

    def one_step(self, pred, feat_id):
        """
            :param pred: prediction.
            :param feat_id: feature id.
            :return: an AXp containing desired feature, or None
        """
        if self.verbose:
            print('(one-step) Feature Membership of d-DNNF into SMT formulas ...')

        ##################### for 0-th replica #####################
        if self.verbose:
            print('Start solving ...')
        time_solving_start = time.process_time()

        ddnnf = self.ddnnf
        # SMT encoding of d-DNNF
        ###############################################################
        # 1. define m feature selectors (s_i), s_0 to s_{nf-1}
        slts = [Symbol(f's_{i}', INT) for i in range(self.nf)]
        # define nf+1 replicas
        cnt = [dict() for _ in range(self.nf+1)]
        dom = dict()
        for i in range(self.nf):
            appear = 0
            for ele in self.lits[i]:
                if ele in self.lit2leaf or -ele in self.lit2leaf:
                    appear += 1
            dom.update({i: appear})
        ###############################################################
        # 2. define counter for each node
        for k in range(self.nf+1):
            if k == 0 or (k > 0 and dom[k-1]):
                for nd in self.dfs_postorder(self.root):
                    if ddnnf.nodes[nd]['label'] == 'F':
                        cnt[k].update({nd: Int(0)})
                    elif ddnnf.nodes[nd]['label'] == 'T':
                        cnt[k].update({nd: Int(1)})
                    else:
                        cnt[k].update({nd: Symbol(f'N_{k}_{nd}', INT)})
        ###############################################################
        # 3. 0 <= s_i <= 1
        tmp0 = []
        for s in slts:
            tmp0.append(And(LE(Int(0), s), LE(s, Int(1))))
        for i in range(self.nf):
            if dom[i] == 0:
                tmp0.append(Equals(slts[i], Int(0)))
        fml_slt_domain = And(tmp0)
        ###############################################################
        # 4. constraints for counter of internal nodes
        tmp1 = []
        for k in range(self.nf+1):
            if k == 0 or (k > 0 and dom[k-1]):
                for nd in self.dfs_postorder(self.root):
                    if ddnnf.nodes[nd]['label'] == 'AND' \
                            or ddnnf.nodes[nd]['label'] == 'OR':
                        tmp_chd = []
                        for chd in ddnnf.successors(nd):
                            tmp_chd.append(cnt[k][chd])
                        if ddnnf.nodes[nd]['label'] == 'AND':
                            tmp1.append(Equals(cnt[k][nd], Times(tmp_chd)))
                        else:
                            tmp1.append(Equals(cnt[k][nd], Plus(tmp_chd)))
        fml_internal = And(tmp1)
        ###############################################################
        # 5. constraints for counter of leaf nodes
        tmp2 = []
        for i in range(self.nf):
            for ele in self.lits[i]:
                if ele in self.lit2leaf:
                    leaf = self.lit2leaf[ele]
                    tmp2.append(Equals(cnt[0][leaf], Int(1)))
                if -ele in self.lit2leaf:
                    nleaf = self.lit2leaf[-ele]
                    tmp2.append(Equals(cnt[0][nleaf], Minus(Int(1), slts[i])))
        for k in range(1, self.nf+1):
            if dom[k-1]:
                for i in range(self.nf):
                    for ele in self.lits[i]:
                        if ele in self.lit2leaf:
                            leaf = self.lit2leaf[ele]
                            tmp2.append(Equals(cnt[k][leaf], Int(1)))
                        if -ele in self.lit2leaf:
                            nleaf = self.lit2leaf[-ele]
                            if i == k-1:
                                tmp2.append(Equals(cnt[k][nleaf], Int(1)))
                            else:
                                tmp2.append(Equals(cnt[k][nleaf], Minus(Int(1), slts[i])))
        fml_leaf = And(tmp2)
        ###############################################################
        # 6. counter of root node
        tmp3 = []
        if pred:
            tmp3.append(Equals(cnt[0][self.root],
                               Times(Minus(Int(2), slts[i]) for i in range(self.nf) if dom[i])))
        else:
            tmp3.append(Equals(cnt[0][self.root], Int(0)))
        fml_out = And(tmp3)
        tmp4 = []
        for i in range(self.nf):
            if dom[i]:
                if pred:
                    tmp4.append(Implies(Equals(slts[i], Int(1)),
                                        LT(cnt[i+1][self.root], Times(Int(2), cnt[0][self.root]))))
                else:
                    tmp4.append(Implies(Equals(slts[i], Int(1)),
                                        LT(Int(0), cnt[i+1][self.root])))
        fml_check_axp = And(tmp4)
        ###############################################################
        # 7. add all constraints
        fml = And(fml_slt_domain, fml_leaf, fml_internal,
                  fml_out, Equals(slts[feat_id], Int(1)), fml_check_axp)
        axp = []
        num_atoms = len(slts) + len(cnt[0])
        for k in range(1, self.nf+1):
            if dom[k-1]:
                num_atoms += len(cnt[k])
        num_fmls = len(tmp0) + len(tmp1) + len(tmp2) + len(tmp3) + len(tmp4) + 1
        with Solver(name="z3") as solver:
            solver.add_assertion(fml)
            if solver.solve():
                val = [solver.get_value(s) for s in slts]
                axp = [i for i in range(self.nf) if val[i] == Int(1)]
                assert feat_id in axp
                axp.sort()
                if self.verbose >= 1:
                    print(f"AXp: {axp}")

        time_solving_end = time.process_time() - time_solving_start
        if self.verbose:
            print(f"Solving time: {time_solving_end:.1f} secs")

        return axp, num_atoms, num_fmls