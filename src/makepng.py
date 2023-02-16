from chromoscalogram import Chromoscalogram as Cscalogram
import sys
import numpy as np
if len(sys.argv) == 1:
    command = sys.argv[0][sys.argv[0].rfind("/")+1:]
    sys.exit(1)

args = sys.argv[1:]
cscal_files = [a for a in args if a.lower().endswith(".cscal")] + [a for a in args if a.lower().endswith(".wav")]


cscals = []
for name in cscal_files: cscals.append(Cscalogram(name))

#ToDo: find largest amplitude among all cscalograms (scale) to lognormalize many
scale= None

for i in range(len(cscals)):
    print("Creating PNG file from \"" +cscal_files[i] + "\".")

    cscals[i].write_to_color_png(scale)