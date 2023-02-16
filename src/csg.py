#Chromoscalogram Class
#To convert from magnitude + phase to complex, magnitude * tan(phase) = complex. if phase > -pi/2 and phase < pi/2, then the real portion is positive
import numpy as np
import sys
import math
import scipy
from scipy import signal 
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
class Chromoscalogram:
    """A class to hold the results of a wavelet transform"""

    BANDWIDTH_FREQ = 100

    def __init__(self, source, name = None):
        """Create a new Chromoscalogram from a WAV or SCAL file, or copy another Chromoscalogram."""
        wav_data = None
        if type(source) == str:

            # open up a WAV file, make a new chromoscalogram
            if source.lower().endswith(".wav"):
                wav_data = Chromoscalogram.extract_wav_data(source)
                self.temp = wav_data[1]
                self.quality = wav_data[0]
                cscal_data = Chromoscalogram.do_transform(wav_data[1], self.quality, 1000, 3 ,25)
                
                self.freqs = cscal_data[0]
                self.data = cscal_data[1]
                
                #name from given name, or the WAV's file name
                if name == None: self.name = source[:-4]
                else: self.name = name

            elif source.lower().endswith(".cscal"):
                
                # get the name
                if name == None: self.name = source[:-5]
                else: self.name = name

                # get the data
                in_stream = open(source, "rb")
                self.quality = int.from_bytes(in_stream.read(4), byteorder="little", signed=False)
                num_freqs = int.from_bytes(in_stream.read(4), byteorder="little", signed=False)
                
                freq_lengths = np.fromfile(in_stream, "<i4", num_freqs)
                
                self.freqs = np.fromfile(in_stream, "<f8", num_freqs)
                
                self.data = []
                for i in range(num_freqs):
                    self.data.append (np.fromfile(in_stream, "<c16", freq_lengths[i]))

                in_stream.close()
            
            else:
                raise TypeError("Cannot generate Chromoscalogram from file " +source+ " of unknown type.")

        elif type(source) == Chromoscalogram:
            if name == None: self.name = source.name
            else: self.name = name
            self.quality = source.quality
            self.freqs = np.copy(source.freqs)
            self.data = np.copy(source.data)

        #final else--we didn't even get a string
        else:
            raise ValueError("Cannot make chromoscalogram: source must be a file name!")
    
    ############################################################################
    # BASIC UTILITIES
    ############################################################################

    def mean(self):
        """Calculates the mean value in this Chromoscalogram."""
        return np.mean(self.data)

    def argmax(self):
        """Where is the maximum value?"""
        index = np.argmax(self.data)
        return (index // self.get_num_freqs(), index % self.get_num_freqs())
    
    def argmin(self):
        """Where is the maximum value?"""
        index = np.argmin(self.data)
        return (index // self.get_num_freqs(), index % self.get_num_freqs())

    def get_num_freqs(self):
        """Return the number of frequencies contained."""
        return len(self.freqs)

    def get_num_times(self):
        """Return the number of time steps contained."""
        return len(self.data[0])

    def __str__(self):
        """Return a string that summarizes the Scalogram's properties."""
        duration = str(self.get_num_times() / self.quality) + " s"
        rate =  "%g kHz" % (self.quality/1000)
        range = "range %g kHz-%g kHz" % ((self.freqs[0]/1000), (self.freqs[-1]/1000))
        num_octaves = math.log(self.freqs[-1] / self.freqs[0], 2)
        voices_per_octave = "%g VpO" % ((self.get_num_freqs()-1) / num_octaves)

        return "<Scalogram \"" +self.name+ "\": " +duration+ ", " +rate+ ", " +range+ ", " +voices_per_octave+ ">"


    ############################################################################
    # SOUND & WAVELET STUFF
    ############################################################################ 

    @staticmethod
    def extract_wav_data(filename):
        """Opens a WAV file, returning a tuple of the quality (sample rate) and the data itself."""

        try:
            import wave
        except ImportError:
            print("Sorry, you need the Python wave module to work with .wav files.", file=sys.stderr)
            sys.exit(1)

        # set up and open file
        w = wave.open(filename)
        num_frames = w.getnframes()
        width = w.getsampwidth()
        quality = w.getframerate()
        data = np.empty(num_frames)
        
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
        return (quality, data)

    @staticmethod
    def make_wavelet(quality, frequency, bandwidth, scaling = 1.0):
        """Make a complex Morlet wavelet"""
        itau = 2.0 * np.pi * (0+1j)
        
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
    
    @staticmethod
    def do_transform(data, quality=44100,  base_freq = 1000, num_octaves = 3, voices_per_octave = 25):
        """Transform .wav data into magnitude and phase over time"""


        # figure out frequencies to test for
        central_freq = base_freq * (2**(num_octaves/2))
        freqs = np.geomspace(base_freq, base_freq * (2**num_octaves), num=num_octaves*voices_per_octave+1, endpoint=True)
        
        # allocation
        cscalogram = []
        

        # do each individual frequency, 1 at a time
        for f in range(len(freqs)):
            scale = central_freq / freqs[f]
            wavelet = Chromoscalogram.make_wavelet(quality, central_freq, 0.000001, scale)
            convolution = np.convolve(data, wavelet,'full')/(central_freq * scale)
            cscalogram.append(convolution)
        return (freqs, cscalogram)

    #@staticmethod
    def do_inverse_transform(self, data, quality=44100, base_freq=1000, num_octaves = 4, voices_per_octave = 25):
        """Transform magnitude and phase over time into waveform data"""

        # frequencies used from transform
        central_freq = base_freq * (2**(num_octaves/2))
        freqs = np.geomspace(base_freq, base_freq * (2**num_octaves), num=num_octaves * voices_per_octave+1, endpoint=True)

        
        # allocation
        temp = Chromoscalogram.make_wavelet(quality, central_freq, 0.000001, central_freq/freqs[0])
        deconvolved = np.zeros(len(data[0]) - len(temp)+1)

        for f in range(len(freqs)):
            
            # prepare each wavelet, and disregard padding for each frequency
            scale = central_freq / freqs[f]
            wavelet = Chromoscalogram.make_wavelet(quality, central_freq, 0.000001, scale)
            
            # add information of each frequency divided by the scale squared
            deconvolved = np.add(np.convolve(data[f], wavelet, 'valid').real/scale/5.5615796, deconvolved)
        #5.5615796 is the evil mystery number. (I currently believe it stems from use of the morlet wavelet in general)
        
        write('test.wav', 44100, deconvolved.astype(np.int16))
        # create .wav file


        #plotting the original and new wav files below
        #x = range(len(deconvolved))
        #fig = plt.figure()
        #ax = fig.add_subplot(1,1,1)
        #ax.plot(x, deconvolved)
        #ax.plot(x, self.temp)
        #ax.set_xlabel('Time')
        #ax.set_ylabel('Magnitude')
        #plt.show()


        return deconvolved
                
    ############################################################################
    # FILE WRITING
    ############################################################################
    def write_to_wav(self, filename=None):
        """Output a WAV of the Chromoscalogram"""
        
        self.do_inverse_transform(self.data, self.quality, base_freq=1000, num_octaves=3, voices_per_octave=25)

    def give_data_to_tf(self):
        """Outputs Chromoscalogram data (magnitude and phase) as np arrays"""
        
        phases = np.zeros([self.get_num_freqs(), self.get_num_times()])
        for f in range(len(self.data)):
            for t in range(len(self.data[f])):
                phases[f][t] = math.atan2(self.data[f][t].imag, self.data[f][t].real).real/math.pi
        
        data = [abs(x) for x in self.data]
        newData = np.empty((2,76,20000))
        for f in range(len(data)):
            padding = (20000-len(phases[f]))//2
            tempMag = np.zeros(20000)
            tempPhase = np.zeros(20000)
            tempPhase[padding:padding+len(phases[f])] = phases[f]
            tempMag[padding:padding+len(data[f])] = data[f]
            
            newData[0][f] = tempMag
            newData[1][f] = tempPhase
             
        return newData

    def write_to_color_png(self, scale=None, filename=None):
        """Output a PNG of the Chromoscalogram"""
        
        phases = np.zeros([self.get_num_freqs(), self.get_num_times()])
        for f in range(len(self.data)):
            for t in range(len(self.data[f])):
                phases[f][t] = math.atan2(self.data[f][t].imag, self.data[f][t].real).real
        
        data = [abs(x) for x in self.data]
        
         
        if scale == None:
            scale = max([max(x) for x in data])
        
        data = [x/scale for x in data]
        toShow = np.zeros([self.get_num_freqs(), self.get_num_times(), 3], dtype=np.uint8)

        for f in range(self.get_num_freqs()):
            padding_left = (len(data[0]) - len(data[f]))//2
            unpadded = 0
            for t in range(padding_left, padding_left + len(data[f])):
                # write each RGB pixel, normalizing between 0-255
                rgb = Chromoscalogram.rgb(phases[f][unpadded])
                # line below presents an alternative coloring/phase representation. A perfect oscillating frequency should be a static color
                #rgb = Chromoscalogram.rgb2(phases[f][unpadded], self.freqs[f], t, self.quality)
                toShow[f][t][0] = (255-rgb[0] * 255 * data[f][unpadded])
                toShow[f][t][1] = (255-rgb[1] * 255 * data[f][unpadded])
                toShow[f][t][2] = (255-rgb[2] * 255 * data[f][unpadded])
                unpadded += 1
        
        toShow = np.flipud(toShow.astype(np.uint8))
        if filename == None: filename = self.name + ".png"

        im = Image.fromarray(toShow)
        im.save(filename, cmin=0, cmax=255)

    def to_numpy_matrix(self, scale=None, filename=None):
        """Output a PNG of the Chromoscalogram"""
        
        phases = np.zeros([self.get_num_freqs(), self.get_num_times()])
        for f in range(len(self.data)):
            for t in range(len(self.data[f])):
                phases[f][t] = math.atan2(self.data[f][t].imag, self.data[f][t].real).real
        
        data = [abs(x) for x in self.data]
        
         
        if scale == None:
            scale = max([max(x) for x in data])
        
        data = [x/scale for x in data]
        toShow = np.zeros([self.get_num_freqs(), self.get_num_times(), 3], dtype=np.uint8)

        for f in range(self.get_num_freqs()):
            padding_left = (len(data[0]) - len(data[f]))//2
            unpadded = 0
            for t in range(padding_left, padding_left + len(data[f])):
                # write each RGB pixel, normalizing between 0-255
                rgb = Chromoscalogram.rgb(phases[f][unpadded])
                # line below presents an alternative coloring/phase representation. A perfect oscillating frequency should be a static color
                #rgb = Chromoscalogram.rgb2(phases[f][unpadded], self.freqs[f], t, self.quality)
                toShow[f][t][0] = (1-rgb[0] * data[f][unpadded])
                toShow[f][t][1] = (1-rgb[1] * data[f][unpadded])
                toShow[f][t][2] = (1-rgb[2] * data[f][unpadded])
                unpadded += 1
        
        toShow = np.flipud(toShow.astype(np.uint8))
        return toShow
 

    def write_to_file(self, filename = None):
        """Writes out a ".cscal" file that contains this Chromoscalogram."""

        # if no file name given, base it off the internally-stored name
        if filename == None:
            filename = self.name + ".cscal"
            #print("using default filename: " +filename)

        # actually write to file, forcing little-endian encoding for everything
        out_stream = open(filename, "wb")

        # first quality & number of freqs, as unsigned 4-byte ints
        out_stream.write((self.quality).to_bytes(4, byteorder="little", signed=False))
        out_stream.write((len(self.freqs)).to_bytes(4, byteorder="little", signed=False))

        # then the freqs array and the amplitudes themselves, as little-endian doubles
        for i in range(len(self.freqs)):
            out_stream.write((len(self.data[i])).to_bytes(4, byteorder="little", signed=True))

        if sys.byteorder == "little":
            for f in self.freqs: out_stream.write(f)
            for row in self.data:
                for d in row: out_stream.write(d)
        else:
            for f in self.freqs: out_stream.write(pack("<d", f))
            for row in self.data:
                for d in row: out_stream.write(pack("<d", d))
        out_stream.close()

    ############################################################################
    # MISCELLANEOUS FUNCTIONS
    ############################################################################
    @staticmethod
    def rgb2(angle, frequency, time, quality):
        """Converts phase data to be one color for each frequency"""
        percentage = (time%(quality/frequency))/(quality/frequency)
        standard_angle = percentage * np.pi * 2
        offset = (angle + math.pi - standard_angle)%(2 * np.pi)
        return Chromoscalogram.rgb(offset)
    

    @staticmethod
    def rgb(angle):
        """Converts a phase angle to color"""
        angle = angle + math.pi
        first = 2 * math.pi/3
        second = 4 * math.pi/3
        # 3 thresholds for the pattern of RGB shifting over ther color wheel
        remainder = angle % first

        # Only two values (R, G, or B) change at a time. They do so on a sin/cos basis, while the unchanging value sits at 0

        calc = remainder/first * math.pi/2

        if angle < first or angle >= math.pi * 2:
            return (math.cos(calc), 0, math.sin(calc))
        elif angle < second:    
            return (0, math.sin(calc), math.cos(calc))
        else:
            return (math.sin(calc), math.cos(calc), 0)