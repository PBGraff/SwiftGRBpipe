from math import log10,sqrt,hypot,atan2
import numpy

# Input data file format:
#0  filename 				ignore this
#1  log_L 					log-luminosity
#2  z 						redshift  [I will log this]
#3  grid_id 				convert to (r,phi) with python func below
#4  bin_size_emit 			source time bin size
#5  alpha 					Band func param
#6  beta 					Band func param
#7  E_peak 					energy peak (log this)
#8  background_name 		name - ignore
#9  bgd_15-25keV 			bkg in band (log this)
#10 bgd_15-50keV 			bkg in band (log this)
#11 bgd25-100keV 			bkg in band (log this)
#12 bgd50-350keV 			bkg in band (log this)
#13 theta 					incoming angle
#14 flux 					flux of burst (log this)
#15 burst_shape 			name - ignore this or turn into feature(s)
#16 ndet 					number of detectors active
#17 rate_GRB_0_global
#18 z1_global
#19 n1_global
#20 n2_global
#21 lum_star_global
#22 x_global
#23 y_global
#24 Epeak_type
#25 Lum_evo_type 
#26 trigger_index			0 if not detected, 1 if detected

class lightcurves:
	names = []
	enc = []
	n = 0
	def read(self,Nenc):
		ifp=open('GRBLCsmooth2_encoded'+repr(Nenc)+'.txt','r')
		enc1=ifp.readlines()
		for i in range(len(enc1)):
			self.names.append(enc1[i].split()[0])
			self.enc.append(enc1[i].split()[1:])
		ifp.close()
		self.n = len(self.names)
	def findidx(self,lcname):
		if self.n>0:
			lcid=self.names.index(lcname)
		else:
			lcid=-1
		return lcid
	def features(self,idx):
		f = numpy.array(self.enc[int(idx)])
		return f

def id2xy(id):
    x=(id%7-3.0)/2.0
    y=(int(id/7)-2.0)/3.0
    r=hypot(x,y)
    phi=atan2(y,x)
    return (r,phi)

def readdata(filepath,Nenc):
    LCs = lightcurves()
    if Nenc>0:
    	LCs.read(Nenc)
    flog=lambda x:log10(float(x))
    conv1={7:flog, 8:len, 9:flog, 10:flog, 11:flog, 12:flog, 14:flog, 15:LCs.findidx}
    conv2={7:flog, 8:len, 9:flog, 10:flog, 11:flog, 12:flog, 14:flog, 15:LCs.findidx, 24:len, 25:len}
    ifp=open(filepath,'r')
    linesplit=ifp.readline().split()
    if len(linesplit)>19:
    	conv=conv2
    	tidx=26
    else:
    	conv=conv1
    	tidx=17
    ifp.close()
    filedata=numpy.loadtxt(filepath,converters=conv,skiprows=1)
    x=numpy.zeros((len(filedata),15+Nenc))
    y=numpy.zeros((len(filedata),))
    x[:,0:2]=filedata[:,1:3]
    x[:,2:4]=map(id2xy,filedata[:,3])
    x[:,4:8]=filedata[:,4:8]
    x[:,8:14]=filedata[:,9:15]
    x[:,14]=filedata[:,16]
    if Nenc>0:
    	x[:,15:(15+Nenc)]=map(LCs.features,filedata[:,15])
    y=filedata[:,tidx]
    return x,y

def PrintPredictions(filename,x,yt,yp,method='forest',sep='\t'):
    (nd,nin) = x.shape
    ofp=open(filename,'w')
    for i in range(nd):
        for j in range(nin):
            ofp.write(repr(x[i][j])+sep)
        if method=='forest':
            ofp.write(repr(yt[i])+sep+repr(yp[i][1])+'\n')
        if method=='svm':
            ofp.write(repr(yt[i])+sep+repr(yp[i])+'\n')
    ofp.close()

def getNames(N):
    names = ['log_L', 'z', 'det_r', 'det_phi', 'bin_size_emit', 'alpha', 'beta', 'E_peak', 'bgd_15-25keV', 'bgd_15-50keV', \
    'bgd25-100keV', 'bgd50-350keV', 'theta', 'flux', 'ndet', 'encLC1', 'encLC2', 'encLC3', 'encLC4', 'encLC5', 'encLC6', \
    'encLC7', 'encLC8', 'encLC9', 'encLC10']
    return names[0:N]
