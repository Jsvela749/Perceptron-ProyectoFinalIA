# Lectura de datos desde el archivo .csv
from neuralNetwork import Perceptron
import pandas, os
from sklearn import preprocessing

def leerCsvNorm(nombreArchivo):
    labels = []
    data = []
    results = []

    datosArchivo = pandas.read_csv(nombreArchivo)
    dataframe = pandas.DataFrame(datosArchivo)

    scaler = preprocessing.StandardScaler()
    dataframe[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']] = scaler.fit_transform(dataframe[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']])

    data = dataframe[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values.tolist()
    results = dataframe['Occupancy'].values.tolist()

    return data, results

# ---------- EJECUCIÓN PRINCIPAL -----------
# Resultados obtenidos de la red neuronal.
results = []
# Errores obtenidos.
errors = []

dataPlot = []

# trainingData, realResults = leerCsv("datatraining.csv")
# Lectura del archivo .csv con datos de entrenamiento normalizados.
trainingData, realResults = leerCsvNorm("datatraining.csv")

# Se crea la neurona con 5 unidades de entrada.
perceptron = Perceptron(5)

# Inicializamos los pesos al azar para las entradas.
perceptron.initializeWeights()

# Entrenamiento de la red.
for i in range(26):
    print(perceptron.weights, "bias =", perceptron.bias)
    errors.clear()
    # if i == 250:
    #     for j in range(len(trainingData)):
    #         print(perceptron.sigmoidFunction(trainingData[j]), realResults[j])
    #     print("")
    for j in range(len(trainingData)):
        result = perceptron.sigmoidFunction(trainingData[j])
        error = perceptron.calculateError(result, realResults[j])
        errors.append(error)
        perceptron.weights[0] += 0.01 * trainingData[j][0] * error
        perceptron.weights[1] += 0.01 * trainingData[j][1] * error
        perceptron.weights[2] += 0.01 * trainingData[j][2] * error
        perceptron.weights[3] += 0.01 * trainingData[j][3] * error
        perceptron.weights[4] += 0.01 * trainingData[j][4] * error
        perceptron.bias += 0.01 * error
    
    dataPlot.append("{} {}".format(i, abs(100 / len(trainingData) * sum(errors))))

    print("El error en la epoca", i, "fue de", abs(100 / len(trainingData) * sum(errors)), "%\n")

# Guardar los pesos conseguidos en un archivo de texto.
weightsFile = open("weights.csv", "w")
for i in range(len(perceptron.weights)):
    weightsFile.write("weight{}, ".format(i))
weightsFile.write("bias\n")
for weight in perceptron.weights:
    weightsFile.write(str(weight) + ", ")
weightsFile.write(str(perceptron.bias))
weightsFile.close()
print("Los pesos y el bias generados se han guardado correctamente en el archivo 'weights.csv'.")

# Preparar el archivo para gráficar con Gnuplot.
plotFile = open("plotFile.gp", "w")
plotFile.write("set xzeroaxis\n")
plotFile.write("set yzeroaxis\n")
plotFile.write("set grid\n")
plotFile.write("set xlabel 'Epocas'\n")
plotFile.write("set ylabel 'Error (En %)'\n")
plotFile.write("plot 'dataPlotFile.txt' with lines linestyle 1\n")
# Guardar los datos obtenidos para generar una gráfica usando GnuPlot.
dataPlotFile = open("dataPlotFile.txt", "w")
for data in dataPlot:
    dataPlotFile.write(data + "\n")
# Cerrando referencias.
dataPlotFile.close()
plotFile.close()
# Llamando al graficador.
os.system("gnuplot -p plotFile.gp")
