"""Code to shuffle bags."""
import math
from random import sample

def shuffle_both(n, bags, y, noise):
    pos_bags = [bag for bag, Yi in zip(bags, y) if Yi]
    neg_bags = [bag for bag, Yi in zip(bags, y) if not Yi]

    n_pos = n
    n_neg = n

    pos_insts = sum(pos_bags, [])
    neg_insts = sum(neg_bags, [])

    pneg = 1.0 - float(len(pos_bags))/float(len(pos_insts))
    pos_size = int(math.ceil(math.log(noise) / math.log(pneg)))
    pos_size = max(1, pos_size)
    pos_size = min(len(pos_insts), pos_size)

    neg_size = int(float(len(neg_insts)) / len(neg_bags))
    neg_size = max(1, neg_size)
    neg_size = min(len(neg_insts), neg_size)

    pos_shuffled = [sample(pos_insts, pos_size) for _ in range(n_pos)]
    neg_shuffled = [sample(neg_insts, neg_size) for _ in range(n_neg)]

    shuffled_bags = pos_shuffled + neg_shuffled
    shuffled_labels = ([True]*n_pos) + ([False]*n_neg)

    return shuffled_bags, shuffled_labels

def shuffle_pos(n, bags, y, noise):
    pos_bags = [bag for bag, Yi in zip(bags, y) if Yi]
    pos_insts = sum(pos_bags, [])

    pneg = 1.0 - float(len(pos_bags))/float(len(pos_insts))
    pos_size = int(math.ceil(math.log(noise) / math.log(pneg)))
    pos_size = max(1, pos_size)
    pos_size = min(len(pos_insts), pos_size)

    pos_shuffled = [sample(pos_insts, pos_size) for _ in range(n)]

    shuffled_bags = pos_shuffled
    shuffled_labels = ([True]*n)

    return shuffled_bags, shuffled_labels
