#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Explaining d-DNNF (OHE) with Anchor
#   Author: Xuanxiang Huang
#
################################################################################
from __future__ import print_function
import time
import sys, csv
import pandas as pd
import numpy as np
from anchor import anchor_tabular
import resource
from math import ceil
from xddnnf.xpddnnf import XpdDnnf

np.random.seed(73)
################################################################################


def anchor_call(model, inst, class_names, feature_names, train_data,
                encoder_fn=None, threshold=0.95, verbose=0):
    classifier_fn = lambda x: pd.Series(model.predict_all(list(x)))

    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_names,
        train_data=train_data)

    feat_sample = np.asarray(inst, dtype=np.float32)

    exp = explainer.explain_instance(feat_sample,
                                     classifier_fn,
                                     threshold=threshold)
    if verbose:
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())

    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
    if verbose:
        print('  time: {0:.2f}'.format(timer))

    expl_set = set(exp.features())
    print(exp.features())
    print(expl_set)
    print(exp.names())
    # assert len(exp.names()) == len(expl_set)

    # length seems incorrect
    return len(expl_set), timer


################################################################################
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            print(f"############ {name} ############")
            ds = pd.read_csv(f"datasets_ohe/{name}.csv")
            df_cleaned = ds.iloc[2:].reset_index(drop=True)

            features = df_cleaned.columns[:-1].tolist()
            feature_to_id = {feature: idx for idx, feature in enumerate(features)}
            label_name = df_cleaned.columns[-1]
            label_values = np.unique(df_cleaned[label_name].values.astype(int))
            X = df_cleaned.iloc[:, :-1].values.astype(int)

            print("Features:", features)
            print("Label Name:", label_name)
            print("X Shape:", X.shape)
            print("Label Values:", label_values)

            data_file = f"examples_ohe/{name}/{name}_inst.csv"
            test_insts_seed = f"samples/test_insts/pmlb_ohe/{name}_seed.csv"
            test_feats = f"samples/test_feats/pmlb_ohe/{name}.csv"
            ddnnf_file = f"examples_ohe/{name}/ddnnf/{name}.dnnf"
            feat_map = f"examples_ohe/{name}/{name}.map"

            ##########
            answer_yes = 0
            atoms_n = []
            fmls_n = []
            calls_n = []
            T_time = []
            seeds = []

            fm_exp = XpdDnnf.from_file(ddnnf_file, verb=1)
            fm_exp.parse_feature_map(feat_map)

            ########### read instance seed file ###########
            with open(test_insts_seed, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    seeds.append(int(line[0]))
            ########### read instance seed file ###########
            ########### read feature file ###########
            with open(test_feats, 'r') as fp:
                feat_lines = fp.readlines()
            ########### read feature file ###########
            # Extract data points using seeds as row indices
            lines = selected_input = X[seeds]
            assert len(lines) == len(feat_lines)

            if len(lines) > 100:
                d_len = 100
            else:
                d_len = len(lines)

            explainer = anchor_tabular.AnchorTabularExplainer(class_names=[0, 1],
                                                              feature_names=features,
                                                              train_data=X)

            for idx, line in enumerate(lines[:d_len]):
                inst = [int(ii) for ii in list(line)]
                fm_exp.parse_instance(inst)
                pred = fm_exp.get_prediction()
                print(f'#{idx}-th instance; prediction: {pred}')

                f_id = int(feat_lines[idx])
                assert 0 <= f_id <= fm_exp.nf - 1

                print(f"{name}, {idx}-th inst file out of {d_len}")

                time_solving_start = time.process_time()

                exp = explainer.explain_instance(np.asarray([inst], dtype=np.int32),
                                                 fm_exp.predict,
                                                 beam_size=20, threshold=0.99)

                feature_indices = exp.features()
                print("Indices of features in explanation:", feature_indices)

                print('Anchor: %s' % (' AND '.join(exp.names())))
                print('Precision: %.2f' % exp.precision())
                print('Coverage: %.2f' % exp.coverage())

                time_solving_end = time.process_time() - time_solving_start

                print('  time: {0:.2f}'.format(time_solving_end))

                if f_id in feature_indices:
                    print('======== Answer Yes ========')
                    answer_yes += 1
                else:
                    print('======== Answer No ========')

                T_time.append(time_solving_end)

            exp_results = f"{name} & {d_len} & "
            exp_results += f"{ceil(answer_yes / d_len * 100):.0f} & "
            exp_results += "{0:.1f} & {1:.1f} & {2:.1f} & {3:.1f}\n" \
                .format(sum(T_time), max(T_time), min(T_time), sum(T_time) / d_len)

            print(exp_results)

            with open('results/anchor/ddnnf_anchor.txt', 'a') as f:
                f.write(exp_results)
