import math
import numpy as np
sig = 4
q = 0.01
delta = 0.00001
T = 10000

eps = math.sqrt(2*np.log(1/delta))/sig
# eps = 1.2
print("basic comp")
print("eps= " , eps , ", delta=" , delta)
print("comp= " , T*eps , ", delta=" , T*delta)


print("advanced comp")
print("eps= " , eps , ", delta=" , delta)
print("comp= " , eps*math.sqrt(T*np.log(1/delta)) , ", delta=" , T*delta)

print("amplification advanced comp")
print("eps= " , eps , ", delta=" , delta)
print("comp= " , 2*eps*q*math.sqrt(T*np.log(1/delta)) , ", delta=" , q*T*delta)

print("moment accountant")
print("eps= " , eps , ", delta=" , delta)
print("2*eps*q=" ,2*eps*q)
print("comp= " , 2*eps*q*math.sqrt(T) , ", delta=" , delta)

