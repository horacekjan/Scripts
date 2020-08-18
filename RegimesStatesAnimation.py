import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from FinticaPlotlib.Graphs import LineGraph, BarGraph, BarLineGraph, ScatterGraph
from FinticaPlotlib.Plotter import Plotter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from openTSNE import TSNE as OpenTSNE


def load_trail(trail_folder, model_name, trail):
    df = pd.read_csv(os.path.join(trail_folder, trail, model_name))
    df = df.set_index('DATE')

    return df.columns, df.values


class TSNEm:
    def __init__(self, n_components=None, random_state=None,
                 initialization="pca", perplexity=30, n_jobs=6):
        self.n_components = n_components
        self.random_state = random_state
        self.tsne = OpenTSNE(n_components=self.n_components,
                             random_state=self.random_state,
                             initialization=initialization,
                             perplexity=perplexity,
                             n_jobs=n_jobs)


    def fit_transform(self, X):
        embeddings = self.tsne.fit(X)
        self.embeddings = embeddings
        return embeddings


    def transform(self, x):
        return self.embeddings.transform(x)


FROM_ITERATION = 1000
TO_ITERATIONS = 1002

model_id = '0.0-model_fcAE~GMM~full~euclidean~mle~empirical~probs~conservative-embedds_24-reg_5-states_45-frwrd_1-lb_6-adapt_None-tovr_cma~10-win_0~1~1-onlr_None-hyp_None-dlevel_0-sd_0-metric_SP500nikko-inputs_nikko~v0-test.csv'
train_model_id = '0.0-model_fcAE~GMM~full~euclidean~mle~empirical~probs~conservative-embedds_24-reg_5-states_45-frwrd_1-lb_6-adapt_None-tovr_cma~10-win_0~1~1-onlr_None-hyp_None-dlevel_0-sd_0-metric_SP500nikko-inputs_nikko~v0-train.csv'
plot_folder = Path(r'C:\Users\Admin\.PyCharm2019.2\config\scratches\Animation')
train_folder = plot_folder / 'train'
test_folder = plot_folder / 'test'
sp500 = pd.read_csv(plot_folder / 'SP500_TR.csv').set_index('date')

trails = ['embeddings', 'regimes_probs', 'states_probs', 'time_series']
names = ['Embeddings', 'Regimes probabilities', 'States probabilities', 'Time series']

regimes = pd.read_csv(test_folder / 'regimes' / model_id, index_col='DATE')
regimes_test_flat = pd.DataFrame([np.where(r == 1)[0][0] for r in regimes.values], index=regimes.index)[0]

regimes_train = pd.read_csv(train_folder / 'regimes' / train_model_id, index_col='DATE')
regimes_train_flat = pd.DataFrame([np.where(r == 1)[0][0] for r in regimes_train.values], index=regimes_train.index)[0]

regimes_num = len(np.unique(regimes_test_flat))

cmap = plt.get_cmap('viridis', regimes_num)
colors = cmap(np.linspace(0, 1, regimes_num))

plotter = Plotter(3, 3)
bar_data = regimes_test_flat[:TO_ITERATIONS]
bar_data = np.roll(bar_data, 1)  # Shift data because of prediction meaning
top_graph = BarLineGraph(sp500.iloc[:TO_ITERATIONS], bar_data=bar_data, title='S&P 500', colors=colors,
                         legend=False)
plotter.add_graph(top_graph, (0, slice(None, None)))
columns, values = load_trail(test_folder / 'trail', model_id, 'regimes_probs')

_min = values.min()
_max = values.max()
values = values[:TO_ITERATIONS]


def normalization(value, _min, _max):
    value = value - _min
    value = value * (1 / (_max - _min))
    value = value / sum(value)
    return value


# values = np.apply_along_axis(functools.partial(normalization, _min=_min, _max=_max), 1, values)

regimes_prob_graph = BarGraph(columns, values, title='Regimes Closeness', colors=colors, y_lim=(_min, _max))
plotter.add_graph(regimes_prob_graph, (1, slice(None, None)))

columns, values = load_trail(test_folder / 'trail', model_id, 'states_probs')
values = values[:TO_ITERATIONS]

# states_prob_graph = BarGraph(columns, values, title='States probabilities')
# plotter.add_graph(states_prob_graph, (slice(1, None), 2))

test_embeddings = pd.read_csv(test_folder / 'trail' / 'embeddings' / model_id)
test_embeddings = test_embeddings.drop('DATE', axis=1)

train_embeddings = pd.read_csv(train_folder / 'trail' / 'embeddings' / train_model_id)
train_embeddings = train_embeddings.drop('DATE', axis=1)

# data = [[] for i in range(FROM_ITERATION)]
# for i in range(FROM_ITERATION, TO_ITERATIONS):
#     print(i)
#     TSNE_embedding = TSNE(n_components=2, perplexity=20, random_state=0, n_jobs=-1).fit_transform(embeddings.iloc[:i])
#     TSNE_embedding = UMAP(n_components=2, n_neighbors=20, random_state=0).fit_transform(embeddings.iloc[:i])
# tsne_df = pd.DataFrame()
# tsne_df['x'] = TSNE_embedding[:, 0]
# tsne_df['y'] = TSNE_embedding[:, 1]
# tsne_df['labels'] = regimes_flat.values[:i]
# data.append(tsne_df)

data = [[] for i in range(FROM_ITERATION)]

tsne = TSNEm(n_components=2, random_state=0, n_jobs=-1)
tsne.fit_transform(train_embeddings.values)

for i in range(FROM_ITERATION, TO_ITERATIONS):
    print(i)
    test_tsne_embeddings = tsne.transform(test_embeddings.values[:i])
    tsne_test_df = pd.DataFrame()
    tsne_test_df['x'] = test_tsne_embeddings[:, 0]
    tsne_test_df['y'] = test_tsne_embeddings[:, 1]
    tsne_test_df['labels'] = regimes_test_flat.values[:i]

    data.append(tsne_test_df)

scatter_graph = ScatterGraph('x', 'y', data, legend=False, hue='labels', s=8, ticks=False,
                             edgecolor="none", colors=colors, title='Regimes Embedding', tracking_last_point=True)

plotter.add_graph(scatter_graph, (2, slice(None, None)))

plotter.save_animation('Images3', FROM_ITERATION)
