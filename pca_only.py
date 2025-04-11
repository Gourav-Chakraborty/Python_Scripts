import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align
import seaborn as sns

u = mda.Universe("pNR_cter_7.5_Mg_all.prmtop", "pNR_cter_7.5_Mg_all.nc")

aligner = align.AlignTraj(u, u, select='name CA',in_memory=True).run()
pc = pca.PCA(u, select='name CA',align=True, mean=None,n_components=None).run()

CA = u.select_atoms('name CA')
n_bb = len(CA)
print('There are {} CA atoms in the analysis'.format(n_bb))
print(pc.p_components.shape)

print(f"PC1: {pc.variance[0]:.5f}")

for i in range(5):
    print(f"Cumulated variance: {pc.cumulated_variance[i]:.3f}")

import matplotlib.font_manager as font_manager
plt.figure(figsize=(12, 8))
plt.plot(pc.cumulated_variance[:50],color='blue', linewidth=2.5)
plt.xlabel("Principal Components", fontsize=20, fontweight='bold', fontname="serif", labelpad=15)
plt.ylabel("Cumulative Variance", fontsize=20, fontweight='bold', fontname="serif", labelpad=15)
plt.xticks(fontsize=15, fontweight='bold', fontname="serif")
plt.yticks(fontsize=15, fontweight='bold', fontname="serif")

# Customize spines and ticks
border_width = 2
for spine in plt.gca().spines.values():
    spine.set_linewidth(border_width)
plt.tick_params(axis='x', which='both', direction='out', length=6, width=2)
plt.tick_params(axis='y', which='both', direction='out', length=6, width=2)

plt.savefig('pNR_cter_7.5_Mg_all_cumulative_variance.png', dpi=150, bbox_inches='tight')

transformed = pc.transform(CA, n_components=5)
transformed.shape
df = pd.DataFrame(transformed,
                  columns=['PC{}'.format(i+1) for i in range(5)])
df['Time (ps)'] = df.index * u.trajectory.dt
df.head()
df.to_csv('pNR_cter_7.5_Mg_all_pca_reduced_data.csv')

for i in range(5):
    cc = pca.cosine_content(transformed, i)
    print(f"Cosine content for PC {i+1} = {cc:.3f}")
    
# melt the dataframe into a tidy format
melted = pd.melt(df, id_vars=["Time (ps)"],
                 var_name="PC",
                 value_name="Value")
g = sns.FacetGrid(melted, col="PC")
g.map(sns.lineplot,
      "Time (ps)", # x-axis
      "Value", # y-axis
      ci=None) # no confidence interval
plt.savefig('pNR_cter_7.5_Mg_all_cosine_content.png', dpi = 150)    
