import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os.path as path




def main():
    fpath = path.join('Data', 'Bikedata.csv')
    with open(fpath, newline=None) as csvfile:
        breader = csv.reader(csvfile,delimiter=',')
        with plt.xkcd():
            x = []
            y = []
            for row in breader:
                ysub = []
                for i in range(1,len(row)):
                    # print(row[i])
                    if (row[i] != '') & (row[i] != '\n'):
                        ysub.append(float(row[i]))
                x.append(float(row[0]))
                y.append(ysub)
            fig = plt.figure()
            font = {'size': 30}
            rc('font',**font)
            ax = []
            for xe, ye in zip(x,y):
                avg = (np.mean(ye))
                ax.append(avg)
                stdev = np.std(ye)
                yl = [avg-stdev,avg+stdev]
                plt.scatter([xe]*len(ye),ye,s=100,c=[1,.5,0],cmap='plasma',linewidths=None,edgecolors='b')
                plt.plot([xe]*2, yl, label='Std Dev', alpha=.6, linewidth=7)

            aplt = plt.scatter(x, ax, s=200, marker="H", alpha=.4, norm=0, edgecolors='b',label='plot1')
            # plt.legend([aplt],['Average'])
            plt.title('IS THE CADENCE STUFF GOOD??')
            plt.annotate("MORE EFFICIENT ALTERNATIVE POSSIBLE.  \nWILL INVOLVE MONKEYS. \n"
                         "AND BANANAS.",
                         xy = (40,100))
            plt.legend([aplt],['Average'])
            plt.xlabel('Observed Cadence')
            plt.ylabel('Calculated Cadence')
            plt.show()
            plt.hold(False)

main()