import numpy as np
import sys
import math
import scipy
from scipy import signal 
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import wave,struct

################### SCALAGRAM CLASS ##########################
class Scalagram:
    def __init__(self,file:str):
        self.name = file.split('.')[-2].split('/')[-1] ## basically just the name of the file w/out the end or the beginning
        self.quality, self.temp = get_wave_data(file)
        self.freqs,self.data = do_transform(self.temp, self.quality, 1000, 3 ,25)
        
    def __str__(self):
        """Return a string that summarizes the Scalogram's properties."""
        duration = str(self.get_num_times() / self.quality) + " s"
        rate =  "%g kHz" % (self.quality/1000)
        range = "range %g kHz-%g kHz" % ((self.freqs[0]/1000), (self.freqs[-1]/1000))
        num_octaves = math.log(self.freqs[-1] / self.freqs[0], 2)
        voices_per_octave = "%g VpO" % ((len(self.freqs)-1) / num_octaves)

        return "<Scalogram \"" +self.name+ "\": " +duration+ ", " +rate+ ", " +range+ ", " +voices_per_octave+ ">"
    def get_data(self, scale=None, filename=None):
        """Output a PNG of the Chromoscalogram"""
        
        phases = np.zeros([len(self.freqs), len(self.data[0])])
        for f in range(len(self.data)):
            for t in range(len(self.data[f])):
                phases[f][t] = math.atan2(self.data[f][t].imag, self.data[f][t].real).real
                
        data = [abs(x) for x in self.data]
        newData = np.empty((2,len(self.freqs),len(phases[0])))
        for f in range(len(data)):
            padding = (len(phases[0])-len(phases[f]))//2
            tempMag = np.zeros(len(phases[0]))
            tempPhase = np.zeros(len(phases[0]))
            tempPhase[padding:padding+len(phases[f])] = phases[f]
            tempMag[padding:padding+len(data[f])] = data[f]
            
            newData[0][f] = tempMag
            newData[1][f] = tempPhase
        
        return newData.transpose()
        
        
################## FUNCTIONS #################################
def make_wavelet(quality, frequency, bandwidth, scaling = 1.0):
        """Make a complex Morlet wavelet"""
       # print(str(quality) + ", " + str(frequency) + ", " + str(bandwidth) + ", " + str(scaling))
        itau = 2.0 * np.pi * (0+1j) # WHere does j come from???
        
        # determine width from bandwidth
        stdev = np.sqrt(bandwidth/2)
        limit = int(np.ceil(stdev * 3 * quality * scaling)) # out to 3 stdevs
        wavelet = np.empty((2*limit+1), dtype=complex)
        z = len(wavelet)//2

        # loop thru every element in half the wavelet
        coefficient = 1.0 / (np.sqrt(np.pi * bandwidth))
        for i in range(z+1):
            t = i / quality / scaling 

            sinusoid = np.exp(t * itau * frequency)
            gaussian = np.exp(-t*t/bandwidth)
            wavelet[z+i] = sinusoid * gaussian
            wavelet[z-i] = wavelet[z+i].conjugate() # complex conjugate for other half
        
        return wavelet

#Transform .wav data into magnitude and phase over time
def do_transform(data, quality=44100,  base_freq = 1000, num_octaves = 3, voices_per_octave = 25):
    central_freq = base_freq * (2**(num_octaves/2))
    
    #spaces numbers evenly on a log scale. start, stop, num values, and whether endpoint is included
    freqs = np.geomspace(base_freq, base_freq * (2**num_octaves), num=num_octaves*voices_per_octave+1, endpoint=True)
    
    cscalogram = []
    
    # do each individual frequency, 1 at a time
    for f in range(len(freqs)):
        scale = central_freq / freqs[f]
        wavelet = make_wavelet(quality, central_freq, 0.000001, scale)
        convolution = np.convolve(data, wavelet,'full')/(central_freq * scale)
        cscalogram.append(convolution)
        
    return freqs, cscalogram
    

def get_wave_data(file:str):
    w = wave.open(file,"rb")
    
    num_frames = w.getnframes()
    width = w.getsampwidth()
    quality = w.getframerate()
    data = np.empty(num_frames)
    
    data2 = np.empty(num_frames)
    
    for i in range(w.getnframes()):
        wavedata = w.readframes(1)
        data1 = struct.unpack("<h", wavedata)
        data2[i] = int(data1[0])
    
################# I'm not sure who's code is right here>??? ###############
    # for 1-byte frames, we need values from 0 to 255
    if width == 1:
        for i in range(num_frames):
            frame = w.readframes(1)
            frame_int = int.from_bytes(frame)
            data[i] = frame_int

    # for 2-byte frames (and maybe others) we have signed values
    else:
        for i in range(num_frames):
            frame = w.readframes(1)
            frame_int = int.from_bytes(frame, byteorder='little', signed = True)
            data[i] = frame_int
        # close file and return the array
    w.close()
    # for i in zip(data,data2):
    #     print(str(i[0]) + " - " + str(i[1]))
    # print("width is: " + str(width)) 
    
    return quality, data2


    
    
################### MAIN #####################################
if __name__ == "__main__":
    files = sys.argv[1:]
    scalagrams = list(map(lambda a: Scalagram(a),files))
    for i in scalagrams[0].get_data().transpose()[1]:
        plt.plot(i)
    plt.show()