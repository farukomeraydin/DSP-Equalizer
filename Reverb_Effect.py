#!/usr/bin/env python
# coding: utf-8

# In[44]:


######################
#Created on Wed May 26
#@author: Berkay Arslan
######################

from scipy.io.wavfile import read, write
from scipy import signal
import numpy as np
from IPython.lib.display import Audio


# In[46]:


# This function is used for avoiding the duplication of reverb sound.
def find_name(file):
    for index in range(len(file)):
        if file[::-1][index] == '.':
            index += 1
            break
    return file[:-index]

# Import Music
audio = wavfile.read("videoplayback.wav")
audio_arr = np.array(audio[1], dtype='float')
read_rate = audio[0]

# Import Impulse
impulse = wavfile.read("French_18th_Century_Salon.wav")
impulse_arr = np.array(impulse[1], dtype='float')
impulse_arr = np.multiply(impulse_arr, 1.0/np.max(impulse_arr))

# Convert Reverb
filtered = signal.convolve(audio_arr, impulse_arr, mode='same', method='fft')
filtered = np.multiply(filtered, 1.0/np.max(np.abs(filtered)))

# Write to the file
if find_name('Reverb_Effect_1.wav') != 'Reverb_Effect_1.wav':
    wavfile.write("Reverb_Effect_1.wav", rate=int(read_rate * 0.9), data=filtered.astype(np.float32))

############ Demonstration #############

# Normal Sound
Fs, x = wavfile.read("videoplayback.wav")
xn = x[:,1]

# Normal Sound with Reverb Effect
Fs_r, xr = wavfile.read("Reverb_Effect_1.wav")
xr_n = xr[:,1]

print('Normal Sound:')
display(Audio(xn, rate=Fs))

print('Normal Sound with Reverb Effect:')
display(Audio(xr_n, rate=Fs_r))


# In[ ]:




