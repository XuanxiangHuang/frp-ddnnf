#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Feature Membership on d-DNNF Classifiers (SAT-based)
#   Author: Xuanxiang Huang
#
################################################################################
import time
import networkx as nx
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from xddnnf.parsing import *
################################################################################


class FmdDnnf(object):
    """
        Deciding FMP for d-DNNF classifier.
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
        #####
        self.vpool = IDPool()
        self.slv = None
        self.slts = []
        self.dom = None
        #####
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

    def new_var(self, name):
        """
            Find or new a PySAT variable.
            See PySat.

            :param name: name of variable
            :return: index of variable
        """
        return self.vpool.id(f'{name}')

    def encode(self, pred):
        """
            SAT encoding.
            :param pred: prediction.
            :return: a weak AXp containing desired feature, or None.
        """
        ###############################################################
        cls = CNF()
        ddnnf = self.ddnnf
        ###############################################################
        slts = [self.new_var(f's_{i}') for i in range(self.nf)]
        cnt = dict()
        dom = dict()
        for i in range(self.nf):
            appear = 0
            for ele in self.lits[i]:
                if ele in self.lit2leaf or -ele in self.lit2leaf:
                    appear += 1
            dom.update({i: appear})
        ###############################################################
        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'F':
                assert False
            elif ddnnf.nodes[nd]['label'] == 'T':
                assert False
            elif ddnnf.nodes[nd]['label'] != 'AND' and ddnnf.nodes[nd]['label'] != 'OR':
                cnt.update({nd: self.new_var(f'L_{nd}')})
            else:
                cnt.update({nd: self.new_var(f'N_{nd}')})
        ###############################################################
        for i in range(self.nf):
            if dom[i] == 0:
                # always universal
                cls.append([-slts[i]])
        ###############################################################
        for nd in self.dfs_postorder(self.root):
            if ddnnf.nodes[nd]['label'] == 'AND' \
                    or ddnnf.nodes[nd]['label'] == 'OR':
                tmp_chd = []
                for chd in ddnnf.successors(nd):
                    tmp_chd.append(cnt[chd])
                if ddnnf.nodes[nd]['label'] == 'AND':
                    for chd in tmp_chd:
                        cls.append([-cnt[nd], chd])
                    cls.append([cnt[nd]] + [-chd for chd in tmp_chd])
                else:
                    for chd in tmp_chd:
                        cls.append([cnt[nd], -chd])
                    cls.append([-cnt[nd]] + tmp_chd)
        ###############################################################
        for i in range(self.nf):
            for ele in self.lits[i]:
                if ele in self.lit2leaf and -ele in self.lit2leaf:
                    leaf = self.lit2leaf[ele]
                    nleaf = self.lit2leaf[-ele]
                    cls.append([cnt[leaf], cnt[nleaf]])
                    cls.append([-cnt[leaf], -cnt[nleaf]])
                if ele in self.lit2leaf:
                    leaf = self.lit2leaf[ele]
                    cls.append([-slts[i], cnt[leaf]])
                if -ele in self.lit2leaf:
                    nleaf = self.lit2leaf[-ele]
                    cls.append([-slts[i], -cnt[nleaf]])
        ###############################################################
        if pred:
            cls.append([-cnt[self.root]])
        else:
            cls.append([cnt[self.root]])
        ###############################################################
        self.slv = Solver(name="Glucose4")
        self.slv.append_formula(cls)
        self.slts = slts
        self.dom = dom

    def is_waxp(self, waxp):
        """
            Checking if a given explanation is a weak AXp.
            :param waxp: given weak AXp
            :return:
        """
        assump = self.slts[:]
        for i in range(self.nf):
            if i not in waxp:
                assump[i] = -assump[i]
        return not self.slv.solve(assumptions=assump)

    def new_pos_claus(self, drop):
        """
            Minimising drop set.
            :param drop: given drop set
        """
        assump = self.slts[:]
        for i in range(self.nf):
            if i in drop or self.dom[i] == 0:
                assump[i] = -assump[i]
        # drop is a weak CXp
        assert self.slv.solve(assumptions=assump)
        step = int(len(drop) / 2)
        for i in range(self.nf):
            if assump[i] < 0 and self.dom[i] > 0 and step > 0:
                # fix free feature
                assump[i] = -assump[i]
                if not self.slv.solve(assumptions=assump):
                    assump[i] = -assump[i]
                step -= 1
        new_drop = []
        for i in range(self.nf):
            if self.dom[i] > 0 and assump[i] < 0:
                new_drop.append(i)
        assert 0 < len(new_drop) <= len(drop)
        return new_drop

    def frp_cegar(self, pred, feat_t):
        """
            SAT encoding.
            :param pred: prediction.
            :param feat_t: target feature.
            :return: a weak AXp containing desired feature, or None.
        """
        if self.verbose:
            print('(cegar) Feature Membership of d-DNNF into SAT formulas ...')
        ###############################################################
        if self.verbose:
            print('Start solving ...')
        time_solving_start = time.process_time()
        ###############################################################
        self.encode(pred)
        waxp = []
        frp_slv = Solver(name="Glucose4")
        slv_calls = 0
        while frp_slv.solve(assumptions=[self.slts[feat_t]]):
            slv_calls += 1
            ##############################
            pick = []
            model = frp_slv.get_model()
            assert model
            for lit in model:
                name = self.vpool.obj(abs(lit)).split(sep='_')
                if name[0] == 's':
                    if lit > 0 and int(name[1]) != feat_t:
                        pick.append(int(name[1]))
            assert feat_t not in pick
            drop = []
            for i in range(self.nf):
                if i == feat_t or self.dom[i] == 0:
                    continue
                if i not in pick:
                    drop.append(i)
            assert feat_t not in drop
            #########################
            if self.is_waxp(pick + [feat_t]):
                # weak axp candidate
                if self.is_waxp(pick):
                    core = self.slv.get_core()
                    new_pick = []
                    for ele in core:
                        if ele in self.slts and ele > 0:
                            new_pick.append(self.slts.index(ele))
                    assert 0 < len(new_pick) <= len(pick)
                    frp_slv.add_clause([-self.slts[i] for i in new_pick])
                else:
                    # waxp is a weak axp
                    waxp = pick + [feat_t]
                    break
            else:
                # pick + feat_t is not a weak axp, add more features
                new_drop = self.new_pos_claus(drop)
                ct_examp = [self.slts[i] for i in new_drop]
                assert ct_examp != []
                frp_slv.add_clause(ct_examp)

        time_solving_end = time.process_time() - time_solving_start
        if self.verbose:
            print(f"Solving time: {time_solving_end:.1f} secs")
        nvars = self.slv.nof_vars()
        nclaus = self.slv.nof_clauses() + frp_slv.nof_clauses()
        self.slv.delete()
        self.slv = None
        frp_slv.delete()
        return waxp, slv_calls, nvars, nclaus, time_solving_end