import pandas as pd
import sys

df = pd.read_csv(sys.stdin, sep='\t', header=None)
pdf = df.pivot_table(columns=1, index=0, values=2)
pdf.to_csv('ckpt_comparison.tsv', sep='\t', index=True)
print(pdf)
