# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:20:50 2024

@author: iosua
"""
import numpy as np
from collections import Counter
import sys

def isValidNumericVector(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()  # Convertir ndarray a lista para que funcione con isinstance
    n = len(x)
    if not all(isinstance(i, (int, float, np.integer, np.floating)) for i in x) or n <= 1:
        return False
    return True

def isValidCategoricalVector(x):
    n = len(x)
    if not all(isinstance(i, (str, bool, np.bool_)) for i in x) or n <= 1:
        return False
    return True

def divideVectorByBreaks(x, breaks, intervalosPosibles, printWarning=False):
  discretized = []
  for value in x:
      closest_i = -1
      for i in range(len(breaks)):
          # Si el valor es menor o igual al break actual
          if value < breaks[i]:
            if i == 0:
                # Si estamos en el primer break, el valor es menor que todos
                discretized.append(intervalosPosibles[0])  # Usar el primer intervalo
            else:
                # Si coincide con el punto de corte
                if value == breaks[i-1]:
                    discretized.append(intervalosPosibles[i])  # Incluir en el intervalo correcto

                else:
                    discretized.append(intervalosPosibles[i])  # Usar el intervalo anterior

            break
          closest_i = i
      if closest_i == len(breaks)-1:
          discretized.append(intervalosPosibles[-1])

  #verificamos que los equal frecuency esten bien
  cantidades = Counter(discretized)
  # Cantdades (frecuencias) de cada valor
  frecuencias = np.array(list(cantidades.values()))
  if (frecuencias.max() - frecuencias.min() >= 2) and printWarning:
      print("Warning: No se ha podido dividir exactamente en intervalos de igual frecuencia debido a valores duplicados en el vector")

  return discretized

def obtenerPosiblesIntervalos(cut_points):
    intervalosDiscretizados = []
    intervalosDiscretizados.append("(-Inf, {})".format(cut_points[0]))
    [intervalosDiscretizados.append("[{}, {})".format(cut_points[i], cut_points[i + 1])) for i in range(len(cut_points) - 1)]
    intervalosDiscretizados.append("[{}, Inf)".format(cut_points[-1]))
    return intervalosDiscretizados

def discretize_EW_base(x, num_bins):
    """Given a vector x and a number of bins, this function
    will separate the numeric vector in num_bins diffent intervals of equal width. That means
    that all intervals will have same length from end to end."""
    n = len(x)
    if not(isValidNumericVector(x)):
        sys.exit('Input vector must be numeric and its length must be > 1')
    if(num_bins>n):
      sys.exit("Error! The number of beans cannot be higher than the length of the vector.")
    elif(num_bins==1):
      sys.exit("Error! The number of bins must be greater than 1.")

    # Calculamos la anchura de cada intervalo de media
    interval_width = (max(x) - min(x)) / num_bins

    # Creamos los puntos de corte
    cut_points = []
    for i in range(1, num_bins):  # Empezamos desde 1 para no incluir el minimo como un cutpoint
        cut_point = min(x) + i * interval_width
        cut_points.append(round(cut_point, 2))

    cut_points =  [round(value,3) for value in cut_points]
    intervalosDiscretizados = obtenerPosiblesIntervalos(cut_points)
    intervalosDiscretizados = np.array(intervalosDiscretizados)
    x_discretized = divideVectorByBreaks(x, cut_points, intervalosDiscretizados)

    return(x_discretized, cut_points)

def discretize_EF_base(x, num_bins):
    """Given a vector x and a number of bins, this functions divides the original vector in
    num_bins intervals, where all intervals contain the same amount of elements in it (+-1 element tolerance).
    This is sometimes not possible when x has duplicated values, so a warning is thrown"""
    n = len(x)
    if not(isValidNumericVector(x)):
        sys.exit('Input vector must be numeric and its length must be > 1')
    if(num_bins>n):
      sys.exit("Error! The number of bins must be lower than the length of the vector.")
    elif(num_bins==1):
      sys.exit("Error! The number of bins must be greater than 1.")

    original_indices = np.argsort(x)
    x = np.sort(x)  # Ordenamos para que sea mas facil calcular los breaks

    elementos_por_bin = n / num_bins

    cut_points = []
    for i in range(num_bins - 1):
        cut_index = int(np.floor((i + 1) * elementos_por_bin))

        # Solo añadimos el punto de corte si es un índice válido
        if cut_index < n:
          if(x[cut_index] in cut_points):#si esta duplicado el punto de corte
            cut_points.append(x[cut_index]+0.01)  # sumarle un pequeño valor
          else:
            cut_points.append(x[cut_index])

    # Redondeado
    cut_points =  [round(value,3) for value in cut_points]
    # Vemos que tenemos todos los cutpoints necesarios
    if len(cut_points) < num_bins - 1:
        cut_points.append(x[-1])  # Añadir el último valor como punto de corte

    intervalosDiscretizados = obtenerPosiblesIntervalos(cut_points)
    x_discretized = divideVectorByBreaks(x, cut_points, intervalosDiscretizados, printWarning=True)
    #volvemos al orden original
    x_discretized_ordered = np.empty(n, dtype=object)
    x_discretized_ordered[original_indices] = x_discretized

    return(x_discretized_ordered.tolist(), cut_points)


def discretize_base(x, cut_points):
    if not(isValidNumericVector(x)):
        sys.exit('Input vector must be numeric and its length must be > 1')
    cut_points = sorted(cut_points)
    intervalosDiscretizados = obtenerPosiblesIntervalos(cut_points)
    x_discretized = divideVectorByBreaks(x, cut_points, intervalosDiscretizados)
    return(x_discretized, cut_points)


def variance_base(x):
    """Function that calculates the variance of the data on given vector x. X must be numeric and has to be
    size > 1"""
    if not(isValidNumericVector(x)):
        sys.exit('Input vector must be numeric and its length must be > 1')
    n = len(x)
    mean_x = sum(x) / n
    sum_squared_diff = sum((xi - mean_x) ** 2 for xi in x)
    variance = sum_squared_diff / (n - 1)
    return variance

def correlation_base(v1, v2):
    """Function to calculate the correlation between two numeric vectors."""
    if not(isValidNumericVector(v1)) or not(isValidNumericVector(v2)):
        sys.exit('Input vector must be numeric and its length must be > 1')
    if len(v1) != len(v2):
        sys.exit('Both vectors must have the same length')
    v1dev = v1 - np.mean(v1)
    v2dev = v2 - np.mean(v2)
    
    # Fórmula de correlación de Pearson
    correlation = np.sum(v1dev * v2dev) / np.sqrt(np.sum(v1dev**2) * np.sum(v2dev**2))
    return correlation

def normalize_base(v):
    if not(isValidNumericVector(v)):
        sys.exit('Input vector must be numeric and its length must be > 1')
    return (v - np.min(v)) / (np.max(v) - np.min(v))

def standarize_base(v):
    if not(isValidNumericVector(v)):
        sys.exit('Input vector must be numeric and its length must be > 1')
    mean_v = np.mean(v)
    variance_v = variance_base(v)
    std_dev_v = np.sqrt(variance_v)  # Calculamos desviacion
    return (v - mean_v) / std_dev_v
    
def entropy_base(v):
    """Calculate the entropy of a numeric vector."""
    if not(isValidCategoricalVector(v)):
        sys.exit('Input vector must be categorical')
    # Contamos frecuencias
    values, counts = np.unique(v, return_counts=True)
    probabilities = counts / len(v)
    # Se calcula la entropía
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def mutualInformation_base(v1, v2):
    """Function to calculate the mutual information between two categorical vectors."""
    if not(isValidCategoricalVector(v1)) or not(isValidCategoricalVector(v2)):
        sys.exit('Input vector must be categorical')
    if len(v1) != len(v2):
        sys.exit('Both vectors must have the same length')
    joint_entropy = entropy_base(np.concatenate((v1, v2)))  # H(X,Y)
    entropy_v1 = entropy_base(v1)
    entropy_v2 = entropy_base(v2)

    # Seguimos la formula I(X, Y) = H(X) + H(Y) - H(X,Y)
    return entropy_v1 + entropy_v2 - joint_entropy



