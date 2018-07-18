"""
    just to play around with the fourier thingy

"""

 #required libraries
import urllib
import scipy.io.wavfile
import pydub
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft as fft
# #a temp folder for downloads
# temp_folder="/Users/home/Desktop/"
# temp_folder="/media/marco/5b508a3b-be31-47a8-a4ba-ef5f30e45caa/Audio_books/"
#
# #spotify mp3 sample file
# web_file="http://p.scdn.co/mp3-preview/35b4ce45af06203992a86fa729d17b1c1f93cac5"
#
# #download file
# urllib.urlretrieve(web_file,temp_folder+"file.mp3")

mpfolder = "/media/marco/5b508a3b-be31-47a8-a4ba-ef5f30e45caa/Audio_books/Alice_in_Wonderland_luisterboek/"
mpfile = "aliceinwonderland_01_carroll_64kb.mp3"

#read mp3 file
mp3 = pydub.AudioSegment.from_mp3(mpfolder+mpfile)
#convert to wav
mp3.export(mpfolder+"converted.wav", format="wav")
#read wav file
rate,audData=scipy.io.wavfile.read(mpfolder+"converted.wav")

print(rate)
print("length of data is {} points".format(len(audData)))

#wav length
print(audData.shape)
print(audData.dtype)
print("data contains {} seconds or {} minutes".format(len(audData) / rate, (len(audData) / rate) / 60))
# energy
print(np.sum(audData.astype(float)**2))
#power - energy per unit of time
print(1.0/(2*(audData.size)+1)*np.sum(audData.astype(float)**2)/rate)





#create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / rate

#plot amplitude (or loudness) over time
# plt.figure(1)
# plt.subplot(211)
# plt.plot(time, audData, linewidth=0.01, alpha=0.7, color='#ff7f00')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.subplot(212)
# plt.plot(time, audData, linewidth=0.01, alpha=0.7, color='#ff7f00')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()


n = int(len(audData))
# print((n+1)/4)
fourier=fft.fft(audData[0:int(n/16)])

fourier = fourier[0:int(len(fourier)/2)]
print("hey")
realStuff = np.real(fourier)
fakeStuff = np.imag(fourier)
#
# plt.plot(fakeStuff, color='red')
# plt.plot(realStuff, color='#ff7f00')
# plt.xlabel('k')
# plt.ylabel('Amplitude')
# plt.show()



# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(len(fourier))

#calculate the frequency at each point in Hz
freqArray = np.arange(0, (int(len(fourier))), 1.0) * (rate*1.0/len(fourier/2));

plt.plot(freqArray/1000, 10*np.log10(fourier), color='#ff7f00', linewidth=0.02)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.show()
