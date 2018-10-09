import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
plotly.tools.set_credentials_file(username='ORANGUH', api_key='T3QLu9WiLRgDLRYz2Fwz')

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy

from scipy import signal

data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')
df = data[0:10]

# table = ff.create_table(df)
# py.iplot(table, filename='wind-data-sample')
#
#
#
# trace1 = go.Scatter(
#     x=list(range(len(list(data['10 Min Std Dev'])))),
#     y=list(data['10 Min Std Dev']),
#     mode='lines',
#     name='Wind Data'
# )
#
# layout = go.Layout(
#     showlegend=True
# )
#
# trace_data = [trace1]
# fig = go.Figure(data=trace_data, layout=layout)
# py.iplot(fig, filename='wind-raw-data-plot')

x=list(range(len(list(data['10 Min Std Dev']))))
y=list(data['10 Min Std Dev'])

plt.plot(x, y)
plt.show()
