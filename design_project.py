from scipy.io import wavfile
from scipy import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

#%% Subroutine to coumpute the smallest value of k_a
def computeKa(N, alpha, ak):
    
    #Calculating (1-alpha)*total power of original signal based on Parseval's theorem 
    #for DFT
    ak = np.absolute(ak)
    ak = np.square(ak)
    val1 = (1-alpha)*np.sum(ak) 
    
    #If only 1 coefficient is needed
    val2 = ak[0]
    if(val2>=val1):
        return 0
    
    #Calculate keep incrementing k_alpha by 1 to see if the power of the 
    #(1-alpha)*original signal is less then the power from -k_alpha to k_alpha
    ka = 0
    for i in range(1,N):
        ka = ka + 1
        val2 = val2 + 2*ak[ka]  #Magnitude of conjugates is the same 
                                #--> add 2(magnitude(a_k))^2
        if(val2 >= val1):       #Check for the condition
            break
    
    return ka

#%% Read the file, calculate the fourier coefficients
fs, x = wavfile.read("Broke_leaf.wav")
x = x/max(x)                    #normalize magnitude to prevent overflow
N = len(x)                      #N=150,000,001

a_k = fft(x)/N                  #a_k = (1/N)*DFT

#%% Compute number of terms for each value of alpha given (0 is added)
alphaArray = np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.005,0.001,0])
minKArray = np.zeros((len(alphaArray),1)) #Parallel array to alphaArray

index = 0
for alpha in alphaArray:
    minKArray[index] = computeKa(N, alphaArray[index], a_k)
    index = index+1

#%% For each value of \alpha and its corresponding ka, store the values from a_0 
#to a_ka in a dictionary
    
alphaDict = {}  #key:alpha, value:values from a_0 to a_ka per alpha
alphaIndex = 0
for alpha in alphaArray:
    ka = int(minKArray[alphaIndex])
    temp = np.zeros(ka, dtype = "complex_") #retains complex numbers
    for index in range(ka):
        temp[index] = a_k[index]
    alphaDict[alpha] = temp
    alphaIndex = alphaIndex + 1

#%% Modifying the Data for Visualization
    
compressionRatio = minKArray/N              #calculating the compression ratio
oneMinusAlpha = np.subtract(1,alphaArray)   #calculating 1-alpha

#%% Plotting (1-\alpha) versus Compression Ratio

plt.figure(1)
plt.plot(oneMinusAlpha, compressionRatio, marker = ".", color = "pink")
plt.ylabel(r"Compression Ratio ($k_\alpha/N$)")
plt.xlabel(r"1-$\alpha$")
#plt.title(r"Compression Ratio versus ($1-\alpha$)")
    
#%% Reconstructing the original signal

reconstructionary = {}                      #key:alpha, value:reconstructed signal
alphaIndex = 0
for alpha in alphaArray:
    coeff = alphaDict[alpha]
    baseifft = ifft(coeff, n=N)
    #Take IFFT of coefficients 0 to ka + conjugate of coefficients from 0 to 
    #ka - (extra coefficient at 0)/N
    temp = (baseifft + np.conjugate(baseifft) - coeff[0]/N)*N 
    reconstructionary[alpha] = temp
    alphaIndex = alphaIndex + 1

#%% Calculate the average power difference between the reconstructed signals versus 
# the original
originalPower = np.sum(np.square(x))/N
powerDiff = np.zeros(len(alphaArray))

index = 0
for alpha in alphaArray:
    reconPower = np.sum(np.square(np.absolute(reconstructionary[alpha])))/N
    powerDiff[index] = originalPower - reconPower
    index = index + 1

#%% plot (1-alpha) versus powerDiff
plt.figure(2)
plt.plot(oneMinusAlpha, powerDiff, marker = ".", color = "black")
plt.ylabel(r"Difference in Power")
plt.xlabel(r"1-$\alpha$")
plt.grid()

#%% Write to wav file to see if this really works...

wavfile.write('Reconstructed_broke_leaf_lossless.wav', fs,np.abs(reconstructionary[0]))

#%% Let's make a 5 point moving average filter to smoothen out the sound and reduce the noise!
    
ma_filtered = np.zeros(N-5)
for i in range(N-5):
    y_ma=0
    for j in range(5):
        y_ma=y_ma+reconstructionary[0][i+j]
    y_ma=y_ma/5
    ma_filtered[i]=y_ma

wavfile.write('5ptmovingaveragefiltered.wav',fs,ma_filtered) #This sounds much better!
