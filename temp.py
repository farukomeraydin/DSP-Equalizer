# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys, wave
import numpy as np
from numpy import array, int16
from scipy.signal import lfilter, butter
from scipy.io.wavfile import read,write
from scipy import signal
from IPython.lib.display import Audio
from scipy.fftpack import fft , ifft, fftshift, ifftshift 
import matplotlib.pyplot as plt
from scipy.io import loadmat

pi = np.pi
gsignal = 0
Fs = 0
xn = 0

def Read_Audio(filename):
    global Fs, xn
    Fs, xn = read(filename)
    xn = convert_to_mono_audio(xn)
    Plot_Spectrum(xn, 10000)
    global gsignal
    gsignal = xn
    print(Fs)
    
def Display_Audio(Fs, xn):
    display(Audio(xn, rate=Fs))
    
def Plot_Spectrum(signal, points):        #Herhangi bir sinyalin arzu edilen noktada fourieri alınıp çizilmesi
    N = points
    w = np.arange(-pi,pi, 2*pi/N)
    xw = fftshift(fft(signal, N)/N)
    plt.figure()
    plt.stem(w/np.pi, abs(xw))
    plt.xlabel('wx$\pi$')
    plt.title('Genlik Grafiği')

def plotWaveform(xn,n,Fs):
    plt.figure(figsize = (16,9))
    plt.plot(n/Fs,xn) # zaman cinsinden belirtmek için n/Fs yapılmıştır.
    plt.xlabel("t")
    plt.ylabel("x(t)")

def convert_to_mono_audio(input_audio):
    output_audio = []
    temp_audio = input_audio.astype(float)
    for i in temp_audio:
        output_audio.append((i[0] / 2) + (i[1] / 2))   #2 sütundaki değerlerin ortalaması alındı ve bunlar tek kanala sokuldu.
    return np.array(output_audio, dtype = 'int16')
        
def set_reverse(input_audio):
    global Fs, xn
    input_audio = input_audio[::-1]  #Satır ve sütunların yeri değiştirilerek ses tersten oynatılacak
    global gsignal
    gsignal = input_audio
    display(Audio(input_audio, rate=Fs))

def set_echo(delay, audio_data):
    output_audio = np.zeros(len(audio_data))
    output_delay = delay * Fs
    for count, i, in enumerate(audio_data):
        output_audio[count] = i + audio_data[count - int(output_delay)]
    audio_data = output_audio
    global gsignal
    gsignal = audio_data
    display(Audio(audio_data, rate=Fs))

def set_speed(speed_factor, xn):
    sound_index = np.round(np.arange(0, len(xn), speed_factor))
    xn = xn[sound_index[sound_index < len(xn)].astype(int)]
    global gsignal
    gsignal = xn
    display(Audio(xn, rate=Fs))

def distortion(x, gain, mix):
    q=x*gain/np.max(np.abs(x));
    z=np.sign(-q)*(1-np.exp(np.sign(-q)*q));
    y=mix*z*np.max(np.abs(x))/np.max(np.abs(z))+(1-mix)*x;
    y=y*np.max(abs(x))/np.max(np.abs(y));
    return y

def basstreble(xn, gain,mode, mix):
    if mix == 1: 
        if mode == 1:
            filter_data = loadmat('filters/bass_lpf.mat') #fc =250Hz
        elif mode == 2:
            filter_data = loadmat('filters/treble_hp.mat') #fc =4000Hz
            
        Coeffs = filter_data['ba'].astype(np.float) 
        b = Coeffs[0,:] 
        yn = lfilter(b, 1,xn*gain)   
        
        return yn     
    else: 
        return xn*gain
    
def equalizer(xn, gain, mode, mix):
    if mix == 1:
        
                        # N = 1000
            
        if mode == 1:   # 20Hz-60Hz-->Sub-bass
            efilter_data = loadmat('filters/equalizer_20Hz-60Hz.mat') 
        elif mode == 2: # 60Hz-200Hz-->Bass
            efilter_data  = loadmat('filters/equalizer_60Hz-250Hz.mat') 
        elif mode == 3: # 200Hz-600Hz-->Lower mids
            efilter_data  = loadmat('filters/equalizer_200Hz-600Hz.mat') 
        elif mode == 4: # 600Hz-3kHz-->Mids
            efilter_data  = loadmat('filters/equalizer_600Hz-3kHz.mat') 
        elif mode == 5: # 3kHz-8kHz-->Upper-Mids
            efilter_data  = loadmat('filters/equalizer_3kHz-8kHz.mat') 
        elif mode == 6: # 8kHz-20kHz-->Highs
            efilter_data  = loadmat('filters/equalizer_8kHz-20kHz.mat') 

            
        Coeffs = efilter_data ['ba'].astype(np.float) 
        b = Coeffs[0,:] 
        yn = lfilter(b, 1,xn)   
        
        return yn*gain     
    else: 
        return xn*gain
    
def find_name(file):
    for index in range(len(file)):
        if file[::-1][index] == '.':
            index += 1
            break
    return file[:-index]

def Save_output(filename):     #Dosya kaydeden fonksiyon
    global Fs
    write(filename, Fs, array(xn, dtype = int16))
    
def reverb(loadname ,xn):
    # Import Music
    audio = read(loadname)
    xn = np.array(audio[1], dtype='float')
    Fs = audio[0]

    # Import Impulse
    impulse = read("French_18th_Century_Salon.wav")
    impulse_arr = np.array(impulse[1], dtype='float')
    impulse_arr = np.multiply(impulse_arr, 1.0/np.max(impulse_arr))

    # Convert Reverb
    filtered = signal.convolve(xn, impulse_arr, mode='same', method='fft')
    filtered = np.multiply(filtered, 1.0/np.max(np.abs(filtered)))

    # Write to the file
    if find_name('Reverb_Effect_1.wav') != 'Reverb_Effect_1.wav':
        write("Reverb_Effect_1.wav", rate=int(Fs), data=filtered.astype(np.float32))

    ############ Demonstration #############

    # Normal Sound
    Fs, x = read("videoplayback.wav")
    xn = x[:,1]

    # Normal Sound with Reverb Effect
    Fs_r, xr = read("Reverb_Effect_1.wav")
    xr_n = xr[:,1]

    print('Normal Sound:')
    display(Audio(xn, rate=Fs))

    print('Normal Sound with Reverb Effect:')
    display(Audio(xr_n, rate=Fs_r))
    
    
ans=True
print('MENU')
while ans:
    print("""
    1. Reverse Effect
    2. Echo Effect
    3. Speed Effect
    4. Distortion Effect
    5. Bass Effect
    6. Treble Effect
    7. Reverb Effect
    8. Equalizer Settings
    9. Exit
    """)
    ans=input("Choose a number: ")
    if ans=="1":
        Read_Audio("videoplayback.wav")
        set_reverse(xn)
    elif ans=="2":
        input_audio = read("videoplayback.wav")
        set_echo(2, input_audio)
    elif ans=="3":
        input_audio = read("videoplayback.wav")
        speed_factor = input("Choose speed factor: ")
        set_speed(speed_factor, input_audio)
    elif ans=="4":
        input_audio = read("videoplayback.wav")
        gain = input("Choose gain: ")
        mix = input("Choose mix type: ")
        distortion(input_audio, gain, mix)       
    elif ans=="5":
        input_audio = read("videoplayback.wav")
        gain = input("Choose gain: ")
        mode = input("Choose mode: ")
        mix = input("Choose mix type: ")
        basstreble(xn, gain, mode, mix)
    elif ans=="6":
       #FIXME
        print('sa')
    elif ans=="7":
        input_audio = read("videoplayback.wav" , xn)
        reverb('videoplayback.wav', )
    elif ans=="8":
        input_audio = read("videoplayback.wav")
        gain = input("Choose gain: ")
        mode = input("Choose mode: ")
        mix = input("Choose mix type: ")    
        equalizer(xn, gain, mode, mix)
    elif ans=="9":
        print("\n Exit") 
        ans = None
    else:
        print("\n Not Valid Choice Try again")
