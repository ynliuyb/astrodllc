from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.rcParams['axes.formatter.limits'] = [-1, 1]
plt.rc('font',family='Times New Roman')
plt.rcParams['font.size'] = 35
fig = plt.figure(figsize=(5, 4), dpi=500)
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'

src_file = ['bbso/bbso_halph_fl_20231201_171442.jpg',
'bbso/bbso_halph_fl_20231201_181444.jpg',
'bbso/bbso_halph_fl_20231201_191449.jpg'
'bbso/bbso_halph_fl_20231201_201436.jpg'
]
df = pd.DataFrame(columns=["(a)", "(b)", "(c)", "(d)"])
image1 = Image.open(src_file[0])
df['(a)'] = np.array(image1).flatten()
image2 = Image.open(src_file[1])
df['(b)'] = np.array(image2).flatten()
image3 = Image.open(src_file[2])
df['(c)'] = np.array(image3).flatten()
image4 = Image.open(src_file[3])
df['(d)'] = np.array(image4).flatten()

correlation_matrix = df.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
np.fill_diagonal(mask, False)

plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_matrix, vmax=1.0, vmin=0.9, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, mask=mask)

x_new = np.arange(0.5, 4.5)
x_label_ticks = ["I-1", "I-2", "I-3", "I-4"]
plt.xticks(x_new, x_label_ticks, rotation=0)
plt.yticks(x_new, x_label_ticks, rotation=0)
plt.subplots_adjust(left=0.07, right=0.992, bottom=0.06, top=0.96)
plt.savefig('bbso-pearson.pdf', dpi=500)
