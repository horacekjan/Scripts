import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from FinticaPlotlib.Graphs import LineGraph, BarGraph, BarLineGraph, ScatterGraph
from FinticaPlotlib.Plotter import Plotter
import matplotlib.pyplot as plt



def load_trail(trail_folder, model_name, trail):
    df = pd.read_csv(os.path.join(trail_folder, trail, model_name))
    df = df.set_index('DATE')

    return df.columns, df.values


FROM_ITERATION = 2
TO_ITERATIONS = 2600

model_id = 'model.csv'
train_model_id = 'model.csv'
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

regimes_prob_graph = BarGraph(columns, values, title='Regimes', colors=colors, y_lim=(_min, _max))
plotter.add_graph(regimes_prob_graph, (1, slice(0, 2)))

columns, values = load_trail(test_folder / 'trail', model_id, 'states_probs')
values = values[:TO_ITERATIONS]
regimes_prob_graph = BarGraph(columns, values, title='States')
plotter.add_graph(regimes_prob_graph, (slice(1, None), 2))

blue = [0.08235294, 0., 0.97254901, 1]

columns, values = load_trail(test_folder / 'trail', model_id, 'embeddings')
values = values[:TO_ITERATIONS]
regimes_prob_graph = BarGraph(columns, values, title='Signatures', colors=[blue for i in range(len(columns))])
plotter.add_graph(regimes_prob_graph, (2, 0))

columns, values = load_trail(test_folder / 'trail', model_id, 'time_series')
values = values[:TO_ITERATIONS]
regimes_prob_graph = BarGraph(columns, values, title='Time series', colors=[blue for i in range(len(columns))])
plotter.add_graph(regimes_prob_graph, (2, 1))

plotter.save_animation('Presentation', FROM_ITERATION)
