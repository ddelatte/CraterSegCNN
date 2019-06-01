from __future__ import print_function
# Module sys has to be imported to get command line arguments
import sys                

# Iteration over all arguments:
#i=0;
#for eachVal in sys.argv:   
#    print(s[i]); i=i+1;
#    print(eachVal)

LINELIMIT = -1

MINLAT = -30
MAXLAT = 0

MINLON = 0 
MAXLON = 30

#s = ['', 'minlat', 'maxlat', 'minlong', 'maxlong']

numLines = 0
#print(sys.argv)

if (len(sys.argv) == 5):
    MINLAT = float(sys.argv[1])
    MAXLAT = float(sys.argv[2])

    MINLON = float(sys.argv[3])
    MAXLON = float(sys.argv[4])
    
    if MAXLAT < MINLAT:
        numLines = -1
    if MAXLON < MINLON:
        numLines = -1


INFILE = "RobbinsCraters_20121016.tsv"

#OUTFILE = "LatLonDiam_RobbinsCraters_20121016.csv"

HAS_HEADER=True

with open(INFILE,'rU') as f:
    if HAS_HEADER:
        line = f.readline()
        print (','.join(str(z) for z in [line.split('\t')[i] for i in [0,1,2,5] ]))
    for line in f:
        if numLines == LINELIMIT:
            break
        
        numLines += 1
        [id, lat, lon, diam] = [line.split('\t')[i] for i in [0,1,2,5] ]
#        if (MINLAT <= float(lat) and float(lat) <= MAXLAT and MINLON <= float(lon) and float(lon) <= MAXLON and MINDIAM <= float(diam) and float(diam) <= MAXDIAM): #includes diameters
        if (MINLAT <= float(lat) and float(lat) <= MAXLAT and MINLON <= float(lon) and float(lon) <= MAXLON): 
        #lat & long only, not diameters
            print (','.join(str(z) for z in [id, lat, lon, diam]))

            

        