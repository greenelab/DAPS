import os

import numpy as np
import seaborn as sns
import pickle as pkl
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

da_classifier = "DA (2)"
comparison_classifier = "SVM"


def load_data(run_name='1'):
    path = './data/' + run_name + '/scores/'
    scores = []
    i = 0

    for file in os.listdir(path):
        if file.endswith('.p'):
            score_dict = pkl.load(open(path + '/' + file, 'rb'))
            missing = 0

            if file.startswith('m'):
                split_file = file.split('_')
                missing = float(split_file[0][1:])
                split_file = split_file[1:]
            else:
                split_file = file.split('_')

            for s in score_dict.keys():
                score = {}
                score['missing'] = missing
                score['model'] = int(split_file[0])
                score['effects'] = int(split_file[1])
                score['per_effect'] = int(split_file[2])
                score['mag'] = float(split_file[3])
                score['observed'] = int(split_file[4])
                score['trial'] = int(split_file[5].split('.')[0])+1
                score['patients'] = s

                for m in score_dict[s]:
                    score[m] = np.mean(score_dict[s][m])

                scores.append(score)

    df = pd.DataFrame(scores)
    print(df.columns)
    df = df.rename(columns={'fullrfc': 'Random Forest',
                            'rfc': 'Limited Random Forest',
                            'nearest_neighbors': 'Nearest Neighbors',
                            'svm': 'SVM',
                            'tree': 'Decision Tree',
                            'da_10000_2_fullrfc': 'DA (2)',
                            'da_10000_4_fullrfc': 'DA (4)',
                            'da_10000_8_fullrfc': 'DA (8)',
                            'da_10000_16_fullrfc': 'DA (16)'
                            })

    df['diff'] = df[da_classifier] - df[comparison_classifier]
    df = df[np.isfinite(df['missing'])]
    return df


def scatter(x, colors):
    palette = np.array(sns.color_palette("hls", 10))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(2):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
