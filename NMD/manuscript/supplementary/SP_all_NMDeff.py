import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

dms = pd.read_csv("/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/data/raw/DMS/SP.csv")

# 1. Filtering
dms = dms[dms.sigma <= 1]
dms = dms[dms.wild_type == "no"]

gene_counts = dms["gene"].value_counts()
genes_to_keep = gene_counts[gene_counts > 49].index
dms = dms[dms.gene.isin(genes_to_keep)]

dms = dms[dms["fitness"] >= -3]
dms = dms[dms["fitness"] <= 3]

# 2. Invert fitness to NMDeff
dms["NMDeff"] = dms["fitness"] * (-1)

# Set up grid size
n_genes = len(genes_to_keep)
n_cols = 12
n_rows = (n_genes // n_cols) + int(n_genes % n_cols > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), sharex=True, sharey=True)
axes = axes.flatten()

# sort genes alphabetically
genes_to_keep = sorted(genes_to_keep)

for i, gene in enumerate(genes_to_keep):
    ax = axes[i]
    gene_df = dms[dms.gene == gene]
    ax.scatter(gene_df.PTCposition, gene_df.NMDeff, s=5, alpha=0.7, color='tab:blue')
    # LOESS fit
    if len(gene_df) > 10:
        loess_fit = lowess(gene_df.NMDeff, gene_df.PTCposition, frac=0.3)
        ax.plot(loess_fit[:, 0], loess_fit[:, 1], color='tab:red', linewidth=1)
    ax.set_title(gene, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('')
    ax.set_ylabel('')

# Remove unused axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.show()