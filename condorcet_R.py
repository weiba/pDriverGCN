import numpy as np
import pandas as pd
from collections import defaultdict
import sys


def candidate_names_from_df(df):
    # return list(np.unique(df.values.flatten()))
    candidate = np.array(list(filter(lambda i: not i is np.nan, df.values.flatten())))
    print(len(list(np.unique(candidate))))
    return list(np.unique(candidate))


def weighted_ranks_from_df(df):
    weighted_ranks = []
    for row in df.values:
        new_row = list(filter(lambda i: not i is np.nan, row))
        weighted_ranks.append((list(new_row), 1))
    return weighted_ranks


def _add_remaining_ranks(d, candidate_name, remaining_ranks, weight):
    for other_candidate_name in remaining_ranks:
        d[candidate_name, other_candidate_name] += weight

def _add_ranks_to_d(d, ranks, weight, unvoted_candidates):
    for i, candidate_name in enumerate(ranks):
        remaining_ranks = ranks[i + 1:] + unvoted_candidates
        _add_remaining_ranks(d, candidate_name, remaining_ranks, weight)

def _compute_d(weighted_ranks, candidate_names):
    """Computes the d array in the Schulze method.

        d[V,W] is the number of voters who prefer candidate V over W.

        We consider unvoted candidates as being ranked less than any
        other candidate voted by the voter.
        """
    d = defaultdict(int)
    for ranks, weight in weighted_ranks:
        unvoted_candidates = list(set(candidate_names) - set(ranks))
        _add_ranks_to_d(d, ranks, weight, unvoted_candidates)
        # _add_ranks_to_d(d, ranks, weight)
    return d

cancerType = sys.argv[1]
df = pd.read_csv('./data/%s/Cancer_List/result/sort_gene.txt' % cancerType, delimiter="\t", index_col=0)
# df = pd.read_csv('./data/%s/NCG_711/result/sort_gene.txt' % cancerType, delimiter="\t", index_col=0)

print(df)
candidate_names = candidate_names_from_df(df)

gene_num = len(candidate_names)
gene_frame = pd.DataFrame(np.zeros([gene_num, gene_num]), index=candidate_names, columns=candidate_names)
# print(gene_frame)
weighted_ranks = weighted_ranks_from_df(df)

d = _compute_d(weighted_ranks, candidate_names)
print(len(d))
for key, value in d.items():
    gene_frame.loc[key[0], key[1]] = value
print(gene_frame)

vote_win = gene_frame.values.sum(axis=1)
vote_loss = gene_frame.values.sum(axis=0)
vote_result = vote_win/(vote_win+vote_loss)
result = pd.DataFrame(vote_result.T, index=candidate_names,columns=["score"])
gene_result = result.sort_values(by="score", ascending=False)
print(gene_result)

gene_result.to_csv('./data/%s/Cancer_List/result/condorcet_gene.txt' % cancerType, sep="\t")
# gene_result.to_csv('./data/%s/NCG_711/result/condorcet_gene.txt' % cancerType, sep="\t")



