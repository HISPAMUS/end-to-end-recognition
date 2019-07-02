from pylab import *

# importar el módulo pyplot
import matplotlib.pyplot as plt

def read(file):
    x = []
    y = []

    lines = open(file, 'r').read().splitlines()

    for line in lines:
        x.append(line.split()[1])
        y.append(float(line.split("SER:")[1].split()[0]))

    return x, y

if __name__ == "__main__":
    x_sin_mod, y_sin_mod = read("../hispamus_model_SER.txt")

    x_con_mod, y_con_mod = read("../pruebaHuspamusLst_SER.txt")

    plt.plot(x_sin_mod, y_sin_mod, 'r-*', x_con_mod, y_con_mod, 'b-.*')
    
    plt.grid()
    plt.legend(('Sin modificaciones', 'Con Modificaciones (iter = 3)'))

    plt.show()