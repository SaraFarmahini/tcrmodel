# The TChard dataset
TChard is a datset for TCR-peptide/-pMHC binding prediction.
It includes more than 500,000 samples, derived from heterogenous sources.
Experiments on this dataset can be found in the GitHub repository: https://github.com/nec-research/tc-hard

## Full dataset
* `ds.csv`: the full TChard dataset.

## Training/test hard splits
`ds.hard-splits/`: it contains the training/test hard splits, obtained with the heuristic described in the paper.

## Hard splits - negative samples from randomization only
`tc-hard/ds.hard-splits/<pep+cdr3b OR pep+cdr3b+cdr3a+mhc>/test/only-sampled-negs/`: the test splits obtained excluding the negative assays, i.e. the negative samples derive only from random recombination of the positive tuples. If you test using these splits, you can use the respective traing splits from:

* `tc-hard/ds.hard-splits/<pep+cdr3b OR pep+cdr3b+cdr3a+mhc>/train/only-sampled-negs/` (which excludes the the negative assays)
* `tc-hard/ds.hard-splits/<pep+cdr3b OR pep+cdr3b+cdr3a+mhc>/train/only-sampled-negs.full/` (which includes the negative assays)

## Hard splits - negative samples from negative assays only
`tc-hard/ds.hard-splits/<pep+cdr3b OR pep+cdr3b+cdr3a+mhc>/test/only-neg-assays/`: the test splits obtained excluding the randomized negative samples, i.e. the negative samples derive only from negative assays. If you test on these splits, you can use traing splits from:

* `tc-hard/ds.hard-splits/<pep+cdr3b OR pep+cdr3b+cdr3a+mhc>/train/only-neg-assays/` (which excludes the the randomized negatives)
* `tc-hard/ds.hard-splits/<pep+cdr3b OR pep+cdr3b+cdr3a+mhc>/train/only-neg-assays.full/` (which includes the randomized negatives)

## Licenses
For the content of this repositoy, we provide a non-commercial license, see LICENSE.txt
Since the samples included in the TChard dataset derive from different datasets, we include a `license` column in the CSV file.
The `license` column specifies from which original dataset a sample comes, and which license applies.
The randomized samples, prosent NaN in the `license` column.