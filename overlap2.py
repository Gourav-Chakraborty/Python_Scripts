import sys
from numpy import mean, std
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# change eps if required
eps = 2

# declaring mean list
mean_list = []
std_list = []
# bin center 
bin1 = []

''''
plots all umbella sampling windows in one plot
usage : python overlap2.py input_file.dat
need python 3.6 or above
Written by: Diship Srivastava
mar 07,2022
'''


print("input file1 :",sys.argv[1])
# input file
ip1 = sys.argv[1]

with open(ip1,"r") as data:
    x1 = []
    for line in data:
        p = line.split()
        x1.append(p[0])

count = 0

for item in x1:
    with open(item,"r") as data:
        count = count +1 # assuming no blank files
        x = []
        y = []
        for line in data:
            if not line.startswith("#"): 
                p = line.split()
                x.append(float(p[0]))
                y.append(float(p[1]))
    fg1 = y
    mean_list.append(mean(fg1))
    std_list.append(std(fg1))

print(f"Read a total of {count} files") # need python 3.6 or above
#print(mean_list)
#print(std_list)
#print(mean_list[0])

# now plotting 
for i in range(count):
    #print(i)
    bin1.append(round(mean_list[i],1))
    #print(f"bin1 : {bin1[i]}")
    x = np.linspace(bin1[i]-eps,bin1[i]+eps,num=10000)
    plot1=plt.plot(x,norm.pdf(x,mean_list[i],std_list[i]))


plt.ylim([0.0,1.3])
plt.title("Umbrella Sampling")
plt.xlabel("CV")
plt.ylabel("Probability Distribution Function")
plt.show()


