#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Parsing feature map file and instance file.
#   Author: Xuanxiang Huang
#
################################################################################


def parse_feature_map(map_file):
    """
        Parsing a file mapping a tuple feature,operator,value to a Boolean literal.
        Format is feature:opterator1value(s)(operator2):literal index (>0 or <0).
        operator can be '=', '!=', set '{}', interval ')(]['.

        :param map_file: e.g. age:=12:1 which means literal x_1 denotes age = 12;
                        e.g. age:{10,11,13}:-2 which means literal -x_2 denotes age in {10,11,13};
                        e.g. age:[12,14):3 which means literal x_3 denotes 12<=age<14.
        :return: number of features, features, domain type, binarized features, literals.
    """
    with open(map_file, 'r') as fp:
        lines = fp.readlines()
    # filtering out comment lines (those that start with '#')
    lines = list(filter(lambda l: (not (l.startswith('#') or l.strip() == '')), lines))

    feats = []
    bfeats = []
    bflits = []
    opt_prefix = ('=', '!=', '{', '[', '(')
    index = 0

    assert (lines[index].strip().startswith('NF:'))
    nf = int((lines[index].strip().split())[1])
    index += 1

    assert (lines[index].startswith('Type:'))
    index += 1

    domtype = lines[index].strip().split(',')
    for ele in domtype:
        assert ele in ('discrete', 'continuous')
    index += 1

    assert (lines[index].startswith('Map:'))
    index += 1

    while index < len(lines):
        feat_opt_val_lit = lines[index].strip().split(sep=':')
        dom = feat_opt_val_lit[:-1][-1]
        assert dom.startswith(opt_prefix)
        if dom.startswith('{'):
            assert dom.endswith('}')
        elif dom.startswith(('[', '(')):
            assert dom.endswith((']', ')'))

        if feat_opt_val_lit[0] not in feats:
            feats.append(feat_opt_val_lit[0])
            bfeats.append([tuple(feat_opt_val_lit[:-1])])
            bflits.append([int(feat_opt_val_lit[-1])])

        else:
            idx = feats.index(feat_opt_val_lit[0])
            bfeats[idx].append(tuple(feat_opt_val_lit[:-1]))
            bflits[idx].append(int(feat_opt_val_lit[-1]))

        index += 1

    assert len(feats) == nf
    assert len(bfeats) == nf
    assert len(bflits) == nf
    assert len(domtype) == nf

    return nf, feats, domtype, bfeats, bflits


def parse_instance(nf, domtype, bfeats, bflits, inst):
    """
        Parsing an instance to a list of literals
        and store. Note that this is MANDATORY
        before explaining an instance.

        :param nf: number of features.
        :param domtype: type of feature domain.
        :param bfeats: binarized features.
        :param bflits: literals.
        :param inst: Given instance, e.g. [0,5,2,3].
        :return: literals of this instance.
    """
    assert (nf == len(inst))
    lits = []
    for j in range(nf):
        blits = []
        if domtype[j] == 'discrete':
            val_j = str(inst[j])
            for jj in range(len(bfeats[j])):
                dom = bfeats[j][jj][1]
                if dom.startswith('='):
                    if val_j == dom[1:]:
                        blits.append(bflits[j][jj])
                    else:
                        blits.append(-bflits[j][jj])
                elif dom.startswith('!='):
                    if val_j != dom[2:]:
                        blits.append(bflits[j][jj])
                    else:
                        blits.append(-bflits[j][jj])
                elif dom.startswith('{'):
                    if val_j in dom[1:-1].split(sep=','):
                        blits.append(bflits[j][jj])
                    else:
                        blits.append(-bflits[j][jj])
        else:
            val_j = float(inst[j])
            for jj in range(len(bfeats[j])):
                bound = bfeats[j][jj][1].split(',')
                lbound = float(bound[0][1:])
                ubound = float(bound[1][:-1])
                if bound[0].startswith('(') and bound[1].endswith(')') and lbound < val_j < ubound:
                    blits.append(bflits[j][jj])
                elif bound[0].startswith('[') and bound[1].endswith(']') and lbound <= val_j <= ubound:
                    blits.append(bflits[j][jj])
                elif bound[0].startswith('(') and bound[1].endswith(']') and lbound < val_j <= ubound:
                    blits.append(bflits[j][jj])
                elif bound[0].startswith('[') and bound[1].endswith(')') and lbound <= val_j < ubound:
                    blits.append(bflits[j][jj])
                else:
                    blits.append(-bflits[j][jj])

        # all literals are consistent.
        for ele in blits:
            assert -ele not in blits
        # no literal occur more than once
        tmp = list(set(blits))
        tmp.sort(key=abs)
        lits.append(tmp)

    assert len(lits) == nf

    return lits
