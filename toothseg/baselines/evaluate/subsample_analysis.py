from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


if __name__ == '__main__':
    df = pd.read_csv('../data/subsample/collected_results/results_TestSet_Subsets.csv')

    cases = ['Normal', 'Third molar', 'Misaligned', 'Implant/Pontic', 'Metal artifact', 'Large FOV']
    # cases = ['Normal', 'Large FOV', 'Misaligned', 'Implant/Pontic', 'Third molar', 'Metal artifact']
    models = ['ToothSeg (ours)', 'ReluNet', 'CuiNet', 'WangNet', 'LiuNet']
    df['idx1'] = df.apply(lambda x: cases.index(x['case']), axis=1)
    df['idx2'] = df.apply(lambda x: models.index(x['model']), axis=1)
    idxs = np.lexsort((df['idx1'].to_list(), df['idx2'].to_list()))
    df = df.iloc[idxs]

    _, axs = plt.subplots(1, 2, figsize=(10, 3.67))
    # for i, y in enumerate(['TP Dice (%)', 'Instance F1 (%)', 'Panoptic Dice (%)']):
    for i, y in enumerate(['Instance Panoptic Dice (%)', 'Multiclass Panoptic Dice (%)']):
        sns.barplot(
            df,
            x=[
                -1.7, -0.7, 0.3, 1.3, 2.3, 3.3,
                7, 8, 9, 10, 11, 12,
                14, 15, 16, 17, 18, 19,
                21, 22, 23, 24, 25, 26,
                28, 29, 30, 31, 32, 33,
            ],
            y=y, 
            hue="case",
            ax=axs[i],
            native_scale=True,
            width=1.0,
        )
        axs[i].set_xticks(
            ticks=[0.5, 9.5, 16.5, 23.5, 30.5],
            labels=models,
        )
        axs[i].set_xlim(-5, 34)
        axs[i].get_legend().remove()
        axs[i].set_ylim(0.75, 1)
        if i == 1:
            axs[i].legend(
                loc='upper right', 
                handletextpad=0.6,
                borderpad=0.4,
                borderaxespad=0.5, 
                labelspacing=0.25,
                fontsize=10,
            )
    
    plt.tight_layout()
    plt.savefig('subsample_analysis.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()