import numpy as np
import pandas as pd

Char = {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13"
    , "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20","chr21","chr22","chrX"}

# Cell={"hES","hIMR"}
ch_number="chr15"
Cell="hES"
PATH = "D:\TAD_DATA\TAD/total." + Cell + ".combined.domain"
domain = pd.read_csv(PATH)
print(domain)