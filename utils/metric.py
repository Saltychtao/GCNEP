from collections import defaultdict


def micro_precision(pred,gold):
    correct = 0
    for p,g in zip(pred,gold):
        if p == g:
            correct += 1
    return correct * 1.0 / len(gold)


def each_precision(pred,gold):
    gold_relation_length = defaultdict(lambda:0)
    tp = defaultdict(lambda:0)
    for p,g in zip(pred,gold):
        gold_relation_length[g] += 1
        if p == g:
            tp[g] += 1
    each_precision = []
    for g in gold_relation_length:
        each_precision.append(tp[g]*1.0/gold_relation_length[g])
    return each_precision


def macro_precision(pred,gold):
    precision = each_precision(pred,gold)
    return sum(precision)/len(precision)
