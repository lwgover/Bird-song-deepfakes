def do_transform(data, quality=44100:int,  base_freq = 1000:int, num_octaves = 3:int, voices_per_octave = 25:int):
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


#===============================================================================================================================================================
if __name__ == "__main__":
    #open the file
    f = open(sys.argv[1], "r")
    f.read()
    f.close()
    transform = do_transform(f)