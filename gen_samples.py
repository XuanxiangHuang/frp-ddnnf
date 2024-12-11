#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Generate tested instances/features
import sys
from Orange.data import Table
import random
import csv
import numpy as np
from frp.ddnnf_smt import FmdDnnf
from pysdd.sdd import Vtree, SddManager, SddNode


def ddnnf_gen_tested_insts(data_name, dataset, num_test, save_dir):
    data = Table(dataset)
    print("Dataset instances:", len(data))
    num_test_ = min(len(data), num_test)
    sample_seed_row = np.array(random.sample(list(range(len(data))), num_test_))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_insts/{save_dir}/{data_name}_seed.csv", 'w') as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


def ddnnf_gen_tested_feats(data_name, dataset, model: FmdDnnf, num_test, save_dir):
    data = Table(dataset)
    print("Dataset instances:", len(data))
    num_test_ = min(len(data), num_test)
    # get support features
    sup_var = set()
    for i in range(model.nf):
        for ele in model.bflits[i]:
            if ele in model.lit2leaf or -ele in model.lit2leaf:
                sup_var.add(i)
    print("Support Features:", len(sup_var), f"out of {model.nf}")
    if num_test_ < len(sup_var):
        sample_seed_row = np.array(random.sample(list(sup_var), num_test_))
    else:
        sample_seed_row = np.array(np.random.choice(list(sup_var), num_test_))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_feats/{save_dir}/{data_name}.csv", 'w') as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


def support_vars(sdd: SddManager):
    all_vars = [_ for _ in sdd.vars]
    nv = len(all_vars)
    sup_vars = [None] * nv

    for i in range(nv):
        lit = all_vars[i].literal
        assert (lit == i + 1)
        neglit = -all_vars[i].literal
        if sdd.is_var_used(lit) or sdd.is_var_used(neglit):
            sup_vars[i] = all_vars[i]
    return sup_vars


def to_lits(sup_vars, inst):
    lits = [None] * len(inst)

    for j in range(len(inst)):
        if sup_vars[j]:
            if int(inst[j]):
                lits[j] = sup_vars[j].literal
            else:
                lits[j] = -sup_vars[j].literal
    return lits


def prediction(root: SddNode, lits):
    out = root
    for item in lits:
        if item:
            out = out.condition(item)
    assert out.is_true() or out.is_false()
    return True if out.is_true() else False


def sdd_gen_tested_insts(sdd_file, vtree_file, circuit, num_test, save_dir):
    name = circuit
    ######################  Pre-processing:  #####################
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    sup_vars = []
    features = []
    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)
    ######################  Pre-processing:  #####################

    tested = set()
    d_len = num_test

    round_i = 0
    while round_i < d_len:
        tmp_sample = []
        for ii in range(tmp_nv):
            tmp_sample.append(random.randint(0, 1))
        while tuple(tmp_sample) in tested:
            tmp_sample = []
            for ii in range(tmp_nv):
                tmp_sample.append(random.randint(0, 1))

        assert tuple(tmp_sample) not in tested

        sample = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                sample.append(tmp_sample[ii])

        lits = to_lits(sup_vars, sample)
        pred = prediction(root, lits)

        tested.add(tuple(tmp_sample))
        round_i += 1

    assert len(tested) == num_test
    data = []
    for item in tested:
        csv_item = list(item)
        assert len(csv_item) == tmp_nv
        sample = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                sample.append(int(csv_item[ii]))

        lits = to_lits(sup_vars, sample)
        pred = prediction(root, lits)
        if pred:
            print(f"prediction: {pred}")
        data.append(csv_item)

    with open(f"samples/test_insts/{save_dir}/{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return


def sdd_gen_tested_feats(sdd_file, vtree_file, circuit, num_test, save_dir):
    name = circuit
    ######################  Pre-processing:  #####################
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    sup_vars = []
    features = []

    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)
    print("Support Features:", nv)
    ######################  Pre-processing:  #####################
    if num_test < nv:
        sample_seed_row = np.array(random.sample(list(range(nv)), num_test))
    else:
        sample_seed_row = np.array(np.random.choice(list(range(nv)), num_test))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_feats/{save_dir}/{name}.csv", "w", newline="") as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 4 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}_list.txt", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            data_file = f"examples_ohe/{name}/{name}_inst.csv"
            if args[4] == '-sdd':
                circuit_sdd = f"examples_ohe/{name}/{name}.txt"
                circuit_sdd_vtree = f"examples_ohe/{name}/{name}_vtree.txt"
                if args[2] == '-inst':
                    sdd_gen_tested_insts(circuit_sdd, circuit_sdd_vtree, name, int(args[3]), bench_name)
                elif args[2] == '-feat':
                    sdd_gen_tested_feats(circuit_sdd, circuit_sdd_vtree, name, int(args[3]), bench_name)
            else:
                if args[2] == '-inst':
                    ddnnf_gen_tested_insts(name, data_file, int(args[3]), bench_name)
                elif args[2] == '-feat':
                    ddnnf_md = FmdDnnf.from_file(f"examples_ohe/{name}/ddnnf/{name}.dnnf")
                    ddnnf_md.parse_feature_map(f"examples_ohe/{name}/{name}.map")
                    ddnnf_gen_tested_feats(name, data_file, ddnnf_md, int(args[3]), bench_name)
            print(name)

    exit(0)