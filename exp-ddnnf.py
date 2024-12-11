#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments (CEGAR based)
import csv
import resource
import sys
from Orange.data import Table
from math import ceil
from xddnnf.xpddnnf import XpdDnnf
from frp.ddnnf_sat import FmdDnnf


def ddnnf_guess_one_axp(data_name, dataset, ddnnf_file, feat_map, insts_seed_file, feats_file):
    sat_axps = []
    atoms_n = []
    fmls_n = []
    calls_n = []
    T_time = []
    seeds = []
    # answer feature membership query
    guess = FmdDnnf.from_file(ddnnf_file, verb=1)
    guess.parse_feature_map(feat_map)
    # extract one AXp
    extor = XpdDnnf.from_file(ddnnf_file, verb=0)
    extor.parse_feature_map(feat_map)
    # get sup variables
    sup_var = set()
    for i in range(guess.nf):
        for ele in guess.bflits[i]:
            if ele in guess.lit2leaf or -ele in guess.lit2leaf:
                sup_var.add(i)

    ########### read instance seed file ###########
    with open(insts_seed_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            seeds.append(int(line[0]))
    ########### read instance seed file ###########
    ########### read feature file ###########
    with open(feats_file, 'r') as fp:
        feat_lines = fp.readlines()
    ########### read feature file ###########
    ########### generate instance ###########
    datatable = Table(dataset)
    inst_data = Table.from_table_rows(datatable, seeds)
    lines = list(inst_data.X)
    ########### generate instance ###########
    assert len(lines) == len(feat_lines)
    if len(lines) > 100:
        d_len = 100
    else:
        d_len = len(lines)
    for idx, line in enumerate(lines[:d_len]):
        inst = [int(ii) for ii in list(line)]
        guess.parse_instance(inst)
        extor.parse_instance(inst)
        pred = extor.get_prediction()
        print(f'#{idx}-th instance; prediction: {pred}')

        f_id = int(feat_lines[idx])
        assert 0 <= f_id <= guess.nf-1
        assert f_id in sup_var

        print(f"{data_name}, {idx}-th inst file out of {d_len}")
        print(f"SAT encoding: query on feature {f_id} out of {guess.nf} features:")
        sat_axp = []
        sat_weakaxp, calls, n_atoms, n_fmls, time_i = guess.frp_cegar(pred, f_id)
        if sat_weakaxp:
            print("Answer Yes")
            fix = [False] * guess.nf
            for ii in sat_weakaxp:
                fix[ii] = True
            sat_axp = extor.find_axp(fix)
            sat_axps.append(sat_axp)
            assert f_id in sat_axp
            print(f"AXp: {sat_axp}")
        else:
            print('======== no AXp exists ========')

        atoms_n.append(n_atoms)
        fmls_n.append(n_fmls)
        calls_n.append(calls)

        print('======== Checking ========')
        axps_enum, cxps_enum = extor.enum_exps()
        if sat_weakaxp:
            assert sat_axp in axps_enum
        else:
            for test_axp in axps_enum:
                assert f_id not in test_axp

        T_time.append(time_i)

    exp_results = f"{data_name} & {d_len} & "
    exp_results += f"{len(sup_var)} & {guess.nn} & "
    exp_results += f"{ceil(len(sat_axps) / d_len * 100):.0f} & "
    exp_results += "{0:.0f} & {1:.0f} & " \
        .format(sum(atoms_n) / d_len, sum(fmls_n) / d_len)
    exp_results += f"{sum(calls_n) / d_len:.0f} & "
    exp_results += "{0:.1f} & {1:.1f} & {2:.1f} & {3:.1f}\n"\
        .format(sum(T_time), max(T_time), min(T_time), sum(T_time) / d_len)

    print(exp_results)

    with open('results/ddnnf_frp_cegar/ddnnf_frp_cegar.txt', 'a') as f:
        f.write(exp_results)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}_list.txt", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            print(f"############ {name} ############")
            data_file = f"examples_ohe/{name}/{name}_inst.csv"
            test_insts_seed = f"samples/test_insts/{bench_name}/{name}_seed.csv"
            test_feats = f"samples/test_feats/{bench_name}/{name}.csv"
            ddnnf = f"examples_ohe/{name}/ddnnf/{name}.dnnf"
            fvmap = f"examples_ohe/{name}/{name}.map"
            ddnnf_guess_one_axp(name, data_file, ddnnf, fvmap, test_insts_seed, test_feats)
    exit(0)
