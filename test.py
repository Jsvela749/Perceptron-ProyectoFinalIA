# Lectura de datos desde el archivo .csv 
import csv
import pandas
from neuralNetwork import Perceptron
from sklearn import preprocessing

# Función que normaliza los datos del archivo .csv leído.
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

# Función para leer archivos .csv.
def leerCsv(nombreArchivo, modo):
    labels = []
    data = []
    results = []

    with open(nombreArchivo, "r") as csvFile:
        fileReader = csv.reader(csvFile)

        # Omitimos la fila de las etiquetas.
        labels = next(fileReader)

        for row in fileReader:
            # Control de la variables.
            if modo == 2:
                *rowData, bias = row
                rowData = list(map(float, rowData))
                bias = float(bias)
                return rowData, bias
            # Sintaxis de desestructuración.
            number, date, *rowData, result = row
                
            # Transformar los datos del CSV a float.
            rowData = list(map(float, rowData))
            # Agregando datos y resultados a los arrays.
            data.append(rowData)
            results.append(float(result))

        # Mostar el número de filas de datos leídas.
        print("Se han leido {} filas de datos del archivo {}".format(fileReader.line_num - 1, nombreArchivo))

    # Se retornan los datos de las columnas y los resultados en arreglos separados.
    return data, results

# Array declarado para acumular los errores.
errors = []

# Selección del archivo csv de prueba.
print("Ingrese el número del conjunto de datos de prueba con el que desea trabajar (1 ó 2): ")
seleccion = int(input())
nombreArchivo = "datatest.csv" if (seleccion == 1) else "datatest2.csv"

print("")
# Se cargan los datos de prueba.
testData, realResults = leerCsvNorm(nombreArchivo)

# Cargar los pesos y el bias generados por el script de entrenamiento.
weights, bias = leerCsv("weights.csv", 2)

# Se inicializa la neurona con los valores ya calculados.
perceptron = Perceptron(5)
perceptron.initializeWeights(weights)
perceptron.bias = bias

# Calculo de los resultados.
#print(perceptron.weights, "bias =", perceptron.bias, "\n")
print("Los resultados dados por el modelo se señalan con y* y los reales con y.\n")
for j in range(len(testData)):
    result = perceptron.sigmoidFunction(testData[j])
    result = perceptron.outputFunction(result)
    error = pow(result - realResults[j], 2)
    if error != 0:
        print("j =", j, "; y* =", result, "; y =", realResults[j])
    errors.append(error)
    
print("\nEl error del algoritmo fue de", 100 / len(testData) * sum(errors), "%\n")

