import numpy
import matplotlib as mpl
import pylab
import scipy.stats
import sys
import cPickle

LConv=197.*10**-9

Eres=lambda mi,mh: mh**2/(2*mi)

g=float(sys.argv[1])

InFPath="./MBData/"
OutFPath="./Output/"
print "g is ", g


def nReT(Y,E,gi,Eres):
    return Y/2.*((E-Eres)/((E-Eres)**2+(0.5*Eres*(gi**2))**2)-(E+Eres)/((E+Eres)**2+(0.5*Eres*(gi**2))**2))

def nImT(Y,E,gi,Eres):
    return Y/2.*(Eres/((E-Eres)**2+(0.5*Eres*gi**2)**2)-Eres/((E+Eres)**2+(0.5*Eres*gi**2)**2))

def OscProbZWA(L,E,Eres,Y,mix=0.45):
 #   print (Y*Eres/(E**2.-Eres**2)*(L/LConv))
    return mix*numpy.sin((Y*Eres/(E**2.-Eres**2)*(L/LConv)))**2


def OscProb3Param( L,E,Eres,Y,g, mix=0.45):
  #  print (nReT(Y,E,g,Eres)*(L/LConv))
    return mix*numpy.sin((nReT(Y,E,g,Eres)*(L/LConv)))**2



#Recipe from https://www-boone.fnal.gov/for_physicists/data_release/lowe/miniboone_may09publicdata_instructions.pdf
def Chi2MBOscillatedNu(Y, Eres, g, mix=0.45):
    NewWeight=[]
    for i in MBSimEvents:
        NewWeight.append(OscProb3Param(i[2]/1e2,i[1]*1e6,Eres,Y,g,mix))
    signal=numpy.histogram(MBSimEvents[:,0],weights=MBSimEvents[:,3]*numpy.array(NewWeight)/float(len(MBSimEvents)),bins=MBBinEdges)[0]
#    print signal
    MBDataFit2by2       = numpy.concatenate([MBCounts,MBMuons])
    MBPredictionFit2by2 = numpy.concatenate([MBBackground+signal,MBMuonPrediction])
    MBResidual2by2      = MBDataFit2by2-MBPredictionFit2by2

#    P3by3 = numpy.concatenate([signal,MBBackground,MBMuonPrediction])
    P3by3 = numpy.concatenate([signal,MBBackground,MBMuonPrediction])
    from numpy import linalg 
    TheCov=numpy.zeros_like(MBCovNu)
    for i in range(MBCovNu.shape[0]):
        for j in range(MBCovNu.shape[1]):
            TheCov[i,j] = MBCovNu[i,j] * P3by3[i]*P3by3[j]
    for i in range(0,len(signal)):
        TheCov[i,i]+=signal[i]

    Block11 = TheCov[0:11,0:11]+TheCov[11:22,11:22]+TheCov[11:22,0:11]+TheCov[0:11,11:22]
    Block22 = TheCov[22:30,22:30]
    Block12 = TheCov[0:11,22:30]+TheCov[11:22,22:30]
    Block21 = TheCov[22:30,0:11]+TheCov[22:30,11:22]

    CovMatrix2by2=numpy.block( [    [Block11, Block12],
                                    [Block21, Block22]  ]   )

    Covinv2by2 = linalg.inv(CovMatrix2by2)

    chi2 = numpy.dot(MBResidual2by2,numpy.dot(Covinv2by2,MBResidual2by2))
    
    return float(chi2)


def Chi2MBOscillatedNuBar(Y, Eres, g,mix=0.45):
    NewWeight=[]
    for i in MBSimEvents_an:
        NewWeight.append(OscProb3Param(i[2]/1e2,i[1]*1e6,Eres,Y,g,mix))
    signal=numpy.histogram(MBSimEvents_an[:,0],weights=MBSimEvents_an[:,3]*numpy.array(NewWeight)/float(len(MBSimEvents_an)),bins=MBBinEdges)[0]
#    print signal
    MBDataFit2by2       = numpy.concatenate([MBCounts_an,MBMuons_an])
    MBPredictionFit2by2 = numpy.concatenate([MBBackground_an+signal,MBMuonPrediction_an])
    MBResidual2by2      = MBDataFit2by2-MBPredictionFit2by2

#    P3by3 = numpy.concatenate([signal,MBBackground,MBMuonPrediction])
    P3by3 = numpy.concatenate([signal,MBBackground_an,MBMuonPrediction_an])
    from numpy import linalg 
    TheCov=numpy.zeros_like(MBCovNu_an)
    for i in range(MBCovNu_an.shape[0]):
        for j in range(MBCovNu_an.shape[1]):
            TheCov[i,j] = MBCovNu_an[i,j] * P3by3[i]*P3by3[j]
    for i in range(0,len(signal)):
        TheCov[i,i]+=signal[i]

    Block11 = TheCov[0:11,0:11]+TheCov[11:22,11:22]+TheCov[11:22,0:11]+TheCov[0:11,11:22]
    Block22 = TheCov[22:30,22:30]
    Block12 = TheCov[0:11,22:30]+TheCov[11:22,22:30]
    Block21 = TheCov[22:30,0:11]+TheCov[22:30,11:22]

    CovMatrix2by2=numpy.block( [    [Block11, Block12],
                                    [Block21, Block22]  ]   )

    Covinv2by2 = linalg.inv(CovMatrix2by2)

    chi2 = numpy.dot(MBResidual2by2,numpy.dot(Covinv2by2,MBResidual2by2))
    
    return float(chi2)


def Chi2MBOscillatedCombined(Y, Eres, g,mix=0.45):
    NewWeight_nu=[]
    NewWeight_nubar=[]
    for i in MBSimEvents:
        NewWeight_nu.append(OscProb3Param(i[2]/1e2,i[1]*1e6,Eres,Y,g,mix))
    for i in MBSimEvents_an:
        NewWeight_nubar.append(OscProb3Param(i[2]/1e2,i[1]*1e6,Eres,Y,g,mix))

    signal_nu=numpy.histogram(MBSimEvents[:,0],weights=MBSimEvents[:,3]*numpy.array(NewWeight_nu)/float(len(MBSimEvents)),bins=MBBinEdges)[0]
    signal_nubar=numpy.histogram(MBSimEvents_an[:,0],weights=MBSimEvents_an[:,3]*numpy.array(NewWeight_nubar)/float(len(MBSimEvents_an)),bins=MBBinEdges)[0]

    MBDataFit2by2_nubar       = numpy.concatenate([MBCounts_an,MBMuons_an])
    MBPredictionFit2by2_nubar = numpy.concatenate([MBBackground_an+signal_nubar,MBMuonPrediction_an])
    MBResidual2by2_nubar      = MBDataFit2by2_nubar-MBPredictionFit2by2_nubar

    MBDataFit2by2_nu          = numpy.concatenate([MBCounts,MBMuons])
    MBPredictionFit2by2_nu    = numpy.concatenate([MBBackground+signal_nu,MBMuonPrediction])
    MBResidual2by2_nu         = MBDataFit2by2_nu-MBPredictionFit2by2_nu

    MBResidual4by4 = numpy.concatenate([MBResidual2by2_nu,MBResidual2by2_nubar])
    
    P6by6 = numpy.concatenate([signal_nu,MBBackground,MBMuonPrediction,signal_nubar,MBBackground_an,MBMuonPrediction_an])
    from numpy import linalg 
    TheCov=numpy.zeros_like(MBCovCombined)
    for i in range(MBCovCombined.shape[0]):
        for j in range(MBCovCombined.shape[1]):
            TheCov[i,j] = MBCovCombined[i,j] * P6by6[i]*P6by6[j]
    for i in range(0,len(signal_nu)):
        TheCov[i,i]+=signal_nu[i]
    for i in range(0,len(signal_nubar)):
        TheCov[i+30,i+30]+=signal_nubar[i]
        
    Block11 = TheCov[0:11,0:11]+TheCov[11:22,11:22]+TheCov[11:22,0:11]+TheCov[0:11,11:22]
    Block22 = TheCov[22:30,22:30]
    Block12 = TheCov[0:11,22:30]+TheCov[11:22,22:30]
    Block21 = TheCov[22:30,0:11]+TheCov[22:30,11:22]

    Block33 = TheCov[30:41,30:41]+TheCov[41:52,41:52]+TheCov[41:52,30:41]+TheCov[30:41,41:52]
    Block44 = TheCov[52:60,52:60]
    Block34 = TheCov[30:41,52:60]+TheCov[41:52,52:60]
    Block43 = TheCov[52:60,30:41]+TheCov[52:60,41:52]

    Block31 = TheCov[30:41,0:11]+TheCov[41:52,11:22]+TheCov[41:52,0:11]+TheCov[30:41,11:22]
    Block42 = TheCov[52:60,22:30]
    Block32 = TheCov[30:41,22:30]+TheCov[41:52,22:30]
    Block41 = TheCov[52:60,0:11]+TheCov[52:60,11:22]

    Block13 = TheCov[0:11,30:41]+TheCov[11:22,41:52]+TheCov[11:22,30:41]+TheCov[0:11,41:52]
    Block24 = TheCov[22:30,52:60]
    Block14 = TheCov[0:11,52:60]+TheCov[11:22,52:60]
    Block23 = TheCov[22:30,30:41]+TheCov[22:30,41:52]

    
    CovMatrix4by4=numpy.block( [    [Block11, Block12, Block13, Block14],
                                    [Block21, Block22, Block23, Block24],
                                    [Block31, Block32, Block33, Block34],
                                    [Block41, Block42, Block43, Block44]  ]   )

    Covinv4by4 = linalg.inv(CovMatrix4by4)

    chi2 = numpy.dot(MBResidual4by4,numpy.dot(Covinv4by4,MBResidual4by4))
    
    return float(chi2)






#Load MiniBooNE data
MBBinEdges=numpy.loadtxt("MBData/miniboone_binboundaries_nue_lowe.txt")

MBCounts=numpy.loadtxt("MBData/miniboone_nuedata_lowe.txt")
MBBackground=numpy.loadtxt("MBData/miniboone_nuebgr_lowe.txt")
MBSimEvents=numpy.loadtxt("MBData/miniboone_numunuefullosc_ntuple.txt")
MBMuons=numpy.loadtxt("MBData/miniboone_numudata.txt")
MBMuonPrediction=numpy.loadtxt("MBData/miniboone_numu.txt")

MBCounts_an=numpy.loadtxt("MBData/miniboone_nuebardata_lowe.txt")
MBBackground_an=numpy.loadtxt("MBData/miniboone_nuebarbgr_lowe.txt")
MBSimEvents_an=numpy.loadtxt("MBData/miniboone_numubarnuebarfullosc_ntuple.txt")
MBMuons_an=numpy.loadtxt("MBData/miniboone_numubardata.txt")
MBMuonPrediction_an=numpy.loadtxt("MBData/miniboone_numubar.txt")

MBCovNu= numpy.loadtxt("MBData/miniboone_full_fractcovmatrix_nu_lowe.txt")
MBCovNu_an= numpy.loadtxt("MBData/miniboone_full_fractcovmatrix_nubar_lowe.txt")
MBCovCombined = numpy.loadtxt("MBData/miniboone_full_fractcovmatrix_combined_lowe.txt")

#MBTransmutedNu=numpy.histogram(MBSimEvents[:,0],weights=MBSimEvents[:,3]*float(len(MBSimEvents)),bins=MBBinEdges)[0]
MBTransmutedNu=numpy.histogram(MBSimEvents[:,0],weights=MBSimEvents[:,3]/float(len(MBSimEvents)),bins=MBBinEdges)[0]
MBTransmutedNu_an=numpy.histogram(MBSimEvents_an[:,0],weights=MBSimEvents_an[:,3]/float(len(MBSimEvents_an)),bins=MBBinEdges)[0]



eres = numpy.arange(1e8, 5e8, 1e7)
yimi = numpy.logspace(-4,-1,40)
gridX, gridY = numpy.meshgrid(eres, yimi)

gridZNuE=[]
for i in range(0,len(yimi)):
    gridZRow=[]
    for j in range(0,len(eres)):
        gridZRow.append(Chi2MBOscillatedNu(yimi[i],eres[j],g,0.45))
    print i
    gridZNuE.append(gridZRow)
minZNuE=min(numpy.array(gridZNuE).flatten())

thefile=open(OutFPath+"/NuE_g_"+str(g)+".pkl",'w')
cPickle.dump(numpy.array(gridZNuE),thefile)
thefile.close()


gridZNuEBar=[]
for i in range(0,len(yimi)):
    gridZRow=[]
    for j in range(0,len(eres)):
        gridZRow.append(Chi2MBOscillatedNuBar(yimi[i],eres[j],g,0.45))
    print i
    gridZNuEBar.append(gridZRow)
minZNuEBar=min(numpy.array(gridZNuEBar).flatten())

thefile=open(OutFPath+"/NuEBar_g_"+str(g)+".pkl",'w')
cPickle.dump(numpy.array(gridZNuEBar),thefile)
thefile.close()

gridZNuCombined=[]
for i in range(0,len(yimi)):
    gridZRowCombined=[]
    for j in range(0,len(eres)):
        gridZRowCombined.append(Chi2MBOscillatedCombined(yimi[i],eres[j],g,0.45))
    print i
    gridZNuCombined.append(gridZRowCombined)

minZCombined=min(numpy.array(gridZNuCombined).flatten())


thefile=open(OutFPath+"/Combined_g_"+str(g)+".pkl",'w')
cPickle.dump(numpy.array(gridZNuCombined),thefile)
thefile.close()



