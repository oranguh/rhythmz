import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
plotly.tools.set_credentials_file(username='ORANGUH', api_key='T3QLu9WiLRgDLRYz2Fwz')

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy
import wave

from scipy import signal
import scipy.io.wavfile

# data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')
# df = data[0:10]

LOWPASSFREQUENCY = 500

rate, data = scipy.io.wavfile.read("/home/murco/iiit_kan_lp/wav/kan_0005.wav", "r")
print(rate)
print(data)
# print(data[0:10])
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

x=list(range(len(list(data))))
y=list(data)

plt.plot(x, y)
# plt.show()
plt.savefig("/home/murco/Music/muzak/old.png")

# fc is the cutoff frequency as a fraction of the sampling rate,
# and b is the transition band also as a function of the sampling rate.
# N must be an odd number in our calculation as well.

# e.g. if we want a 500 Hz filter. We need to know our sampling rate for the data
# Say our data has a sampling rate of 44,100 Hz. 500/44100 is 0.01133786848



fc = LOWPASSFREQUENCY/rate
print(fc)
b = 0.08
N = int(np.ceil((4 / b)))
if not N % 2: N += 1
n = np.arange(N)

sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
sinc_func = sinc_func * window
sinc_func = sinc_func / np.sum(sinc_func)

s = list(data)
new_signal = np.convolve(s, sinc_func)



x=list(range(len(new_signal)))
y=list(new_signal)

plt.figure()
plt.plot(x, y)
plt.savefig("/home/murco/Music/muzak/new.png")


scipy.io.wavfile.write("/home/murco/Music/muzak/something.wav", rate, new_signal)

#
# trace1 = go.Scatter(
#     x=list(range(len(new_signal))),
#     y=new_signal,
#     mode='lines',
#     name='Low-Pass Filter',
#     marker=dict(
#         color='#C54C82'
#     )
# )
#
# layout = go.Layout(
#     title='Low-Pass Filter',
#     showlegend=True
# )
#
# trace_data = [trace1]
# fig = go.Figure(data=trace_data, layout=layout)
# py.iplot(fig, filename='fft-low-pass-filter')
