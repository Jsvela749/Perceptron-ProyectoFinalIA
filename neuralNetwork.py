import numpy, random

class Perceptron:
    # Constructor.
    def __init__(self, inputsNumber):
        self.inputsNumber = inputsNumber
        self.bias = 1
        self.weights = [0] * inputsNumber

    # Función que inicializa los pesos para la red neuronal.
    def initializeWeights(self, weights=[]):
        if not weights:
            for i in range(len(self.weights)):
                self.weights[i] = random.uniform(-1, 1, )
        else:
            for i in range(len(self.weights)):
                self.weights[i] = weights[i]

    # Función de activación (sigmoidal).
    def sigmoidFunction(self, inputs):
        result = numpy.dot(inputs, self.weights) + self.bias
        return 1 / (1 + numpy.exp(-result))
    
    def outputFunction(self, inputData):
        if inputData >= 0.5:
            return 1
        else:
            return 0

    # Función que cálcula el error.
    def calculateError(self, output, expected):
        error = output * (1 - output) * (expected - output)
        return error