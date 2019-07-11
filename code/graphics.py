import matplotlib.pyplot as plt

def print_plt(file, format, color):
    x = []
    y = []

    lines = open(file, 'r').read().splitlines()

    for line in lines:
        x.append(line.split()[1])
        y.append(float(line.split("SER:")[1].split()[0]))

    plt.plot(x, y, format, color = color)

if __name__ == "__main__":
    legends = []
    lines = open("resultados/resultados.lst", 'r').read().splitlines()

    for line in lines:
        data = line.split('|')
        print_plt(data[0], data[1], data[2])
        legends.append(data[3])
    
    plt.grid()
    plt.ylim(10, 30)
    plt.xlabel('Epoch')
    plt.ylabel('SER')
    plt.legend(legends)

    plt.show()