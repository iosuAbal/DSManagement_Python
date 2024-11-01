# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:22:10 2024

@author: Iosu Abal
"""
import numpy as np 
import sys
import pandas as pd
from . import myUtils as ut

class Atributo:
    """This class represents an attribute or a column of a dataset."""
    
    def __init__(self, nombre, valores, cortes=None):
        """Constructor of class Atributo"""
        self.nombre = nombre
        # Validar el tipo de valores
        if all(isinstance(i, (bool, np.bool_)) for i in valores):  
            self.tipo = "logical"
        elif ut.isValidNumericVector(valores):  
            self.tipo = "numeric"
        elif ut.isValidCategoricalVector(valores):  
            self.tipo = "character"
        else:
            print(type(valores))
            sys.exit("Attribute values must be numerical, categorical, or logical.")
        
        self.valores = valores
        self.cortes = cortes

    def discretize_EW(self, num_bins):
        """Discretize using equal width. If the attribute is not numeric an error is thrown
        Returns:
            A new Atributo, where all its values are discretized"""
        result = ut.discretize_EW_base(self.valores, num_bins) #valores discrtetizados y cortes
        return Atributo(self.nombre+'  discretized (EW)', result[0], result[1]) #creamos un atributo con valores discretizados y cortes

    def discretize_EF(self, num_bins):
        """Discretize attribute values using equal frequency. If the attribute is not numeric an error is thrown
        Returns:
            A new Atributo, where all its values are discretized"""
        result = ut.discretize_EF_base(self.valores, num_bins) #valores discrtetizados y cortes
        return Atributo(self.nombre+'_discretized(EF)', result[0], result[1]) #creamos un atributo con valores discretizados y cortes
    
    def variance(self):
        """Calculate the variance of the values of the attribute
        Returns:
            The variance"""
        return ut.variance_base(self.valores)

    def entropy(self):
        """Calculate the entropy of the values of the attribute
        Returns:
            The entropy"""
        return ut.entropy_base(self.valores)

    def standarize(self):
        """Function to standarize values of an atribute.
        Returns: 
            A new Atributo with its values standarized (mean 0)"""
        return Atributo(self.nombre+'_standarized', ut.standarize_base(self.valores)) # esta función ya verifica el tipo numerico
    
    def normalize(self):
        """Function to normalize values of an atribute.
        Returns: 
            A new Atributo with its values normalized (between 0 and 1)"""
        return Atributo(self.nombre+'_normalized', ut.normalize_base(self.valores))
        
    def print(self):
        """Print the Atributo in a proper manner"""
        print('Atribute: ',self.nombre,'\nValues:', self.valores, '\nType:',  self.tipo,
              '\nCut points:',  self.cortes)

class Dataset:
    """This class simulates a Dataframe, and will have a list of Atributos"""

    def __init__(self, nombre, atributos):
        """Constructor of class Dataset"""
        self.nombre = nombre
        self.atributos = atributos

    def discretize_EW(self, num_bins):
        """Discretize numeric attributes in the dataset using equal width. That means that all intervals are
        the same size.
        Parameters: 
            num_bins: The number of intervals that the numeric values should be discretized into
        Returns :
            dataset: A new Dataset object, which will have numeric values discretized"""
        results = []
        for attr in self.atributos:
            if attr.tipo == "numeric":
                # Llamamos a discretized_EW de la clase Atributo
                discretized_attr = attr.discretize_EW(num_bins)
                results.append(discretized_attr)
            else:
                results.append(attr)
        return Dataset(self.nombre, results)
        
    def discretize_EF(self, num_bins):
        """Discretize numeric attributes in the dataset using equal-frequency. Other attributes remain exactly the same
        Parameters: 
            num_bins: The number of intervals that the numeric values should be discretized into
        Returns :
            dataset: A new Dataset object, which will have numeric attributes discretized"""
        results = []
        for attr in self.atributos:
            if attr.tipo == "numeric":
                # Llamamos a discretized_EF de la clase Atributo
                discretized_attr = attr.discretize_EF(num_bins)
                results.append(discretized_attr)
            else:
                results.append(attr)
        return Dataset(self.nombre, results)
        
    def calculate_rates(self, numeric_attr_name):
        """Calculate True Positive Rate and False Positive Rate of an attribute of the dataset
        Parameters:
            numeric_attr_name: Name of the numeric attribute that will be taken into account to make the classifier
        Returns:
            dataset: A new Dataset object, which will have numeric attributes, TPR and FPR respectively"""
        numeric_attr = next((attr for attr in self.atributos if attr.nombre == numeric_attr_name), None)

        if numeric_attr is None or numeric_attr.tipo not in ["numeric", "integer"]:
            raise sys.exit("Numeric input attribute not found in dataset.")
        
        # Coger el primer atributo lógico que exista
        logical_attrs = [attr for attr in self.atributos if attr.tipo == "logical"]
        if not logical_attrs:
            raise sys.exit("No logical attribute found")
        logical_attr = logical_attrs[0]
        # Ordenamos
        sorted_indices = np.argsort(numeric_attr.valores)
        valores_logicos = np.array(logical_attr.valores)[sorted_indices]
        n = len(valores_logicos)
        # Calculamos el TPR y FPR para cada cutpoint
        tpr_values = []
        fpr_values = []
        for cut_point in range(n):
            predictions = np.array([False] * cut_point + [True] * (n - cut_point))
            TP = np.sum(predictions & valores_logicos)
            TN = np.sum(~predictions & ~valores_logicos)
            FP = np.sum(predictions & ~valores_logicos)
            FN = np.sum(~predictions & valores_logicos)
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            tpr_values.append(TPR)
            fpr_values.append(FPR)

        tpr_attr = Atributo(nombre="TPR", valores=tpr_values)
        fpr_attr = Atributo(nombre="FPR", valores=fpr_values)
        # Devolvemos un objeto de nuestro tipo Dataset
        return Dataset("Rates",[tpr_attr, fpr_attr])

    
    def calculate_metrics(self):
        """
        Calculate metrics of a dataset: Variance and AUC for each numeric attribute, and entropy 
        for every categorical atribute.
        
        Parameters:
        dataset (Dataset): The dataset object containing attributes.
    
        Returns:
            dictionary: A dictionary with attribute names as keys and the metrics as values. If the attribute is numeric
            entropy will be None and viceversa.
        """
        # Vemos si existe algun atributo logico para calcular el auc
        calc_auc = any(attr.tipo == "logical" for attr in self.atributos)
        if not calc_auc:
            print("Warning: No logical attribute found for calculating AUC.")
        results = {}
        for atributo in self.atributos:
            resultado = {"Variance": None, "AUC": None, "Entropy": None}
    
            if atributo.tipo == "numeric":
                # Varianza (redondeada a 3 decimales)
                resultado["Variance"] = round(atributo.variance(),3)
                # AUC
                if calc_auc:
                    dataset_rates = self.calculate_rates(atributo.nombre)  #nuestra función
                    #cogemos los dos primeros atributos y los convertimos a array con numpy
                    TPR = np.array(next(filter(lambda a: a.nombre == "TPR", dataset_rates.atributos)).valores)
                    FPR = np.array(next(filter(lambda a: a.nombre == "FPR", dataset_rates.atributos)).valores)
                    orden = np.argsort(FPR)
                    FPR = FPR[orden]
                    TPR = TPR[orden]
                    # Aplicamos regla del trapecio para calcular el area bajo la curva
                    auc = np.sum(np.diff(FPR) * (TPR[:-1] + TPR[1:]) / 2)
                    resultado["AUC"] = round(auc,3)
            else:
                # Entropia (atributo no numerico)
                resultado["Entropy"] = round(atributo.entropy(),3)
            results[atributo.nombre] = resultado
        return results

    def standarize(self):
        """Standarize numeric attributes in the dataset.
        Returns: 
            A new Dataset object, where its numeric values are standarized. Other attributes remain exactly the same
        """
        results = []
        for attr in self.atributos:
            if attr.tipo == "numeric":
                st_attr = attr.standarize()
                results.append(st_attr)
            else:
                results.append(attr)
        return Dataset(self.nombre, results)

    def normalize(self):
        """Normalize numeric attributes in the dataset
        Returns: 
            A new Dataset object, where its numeric values are normalized. Other attributes remain exactly the same
        """
        results = []
        for attr in self.atributos:
            if attr.tipo == "numeric":
                # Llamamos a discretized_ew de la clase Atributo
                norm_attr = attr.normalize()
                results.append(norm_attr)
            else:
                results.append(attr)
        return Dataset(self.nombre, results)

    
    def filter_metrics(self, metric_name, operator, value):
        """Filter a dataset based on its metrics of a dataset:
        
        Parameters:
        metric_name: Name of the metric to be filtered by. Must be Variance, AUC or entropy
        operator : Logical condition to be checked: <,>,=,...
        value : The value to which the metric is compared
        
        Returns:
        dataset: A new Dataset object, that just contains the attributes that fulfill the condition"""
        
        if metric_name not in('Variance', 'AUC', 'Entropy'):
            sys.exit('Metric must be Variance, AUC or Entropy')
        metrics_dictionary = self.calculate_metrics()  # Funcion definida arriba
        filtered_metrics = []
        for attribute in self.atributos:
            metric_value = metrics_dictionary[attribute.nombre].get(metric_name)
            if metric_value is not None:  # Comprobar que no sea nulo
                # Mapa de funciones para los operadores
                operator_func = {
                        ">": lambda x: x > value,
                        "<": lambda x: x < value,
                        ">=": lambda x: x >= value,
                        "<=": lambda x: x <= value,
                        "==": lambda x: x == value,
                        "!=": lambda x: x != value,
                }
                # Aplicar la condicion
                if operator in operator_func:
                    filtered_metrics.append(operator_func[operator](metric_value))
                else:
                    raise ValueError(operator, " is not a valid operator")
            else:
                filtered_metrics.append(False) 
    
        # Devolvemos en un dataset solamente los atributos que han cumplido la condicion
        valid_attributes = [attr for attr, valid in zip(self.atributos, filtered_metrics) if valid]
        return Dataset(self.nombre+" filtered", valid_attributes)

    def correlation(self):
        """Calculate the correlation between each pair of attributes of the dataset. if both are numeric, the correlation
        is calculated. In the case that none of them are numeric, the mutual information is computed.
        Returns:
            dictionary: A dictionary that will have two dataframes inside, one for the correlation pairs, and other for
                        mutual information attributes."""
        numeric_indices = []
        categorical_indices = []
        num_atributos = len(self.atributos)
        result = np.full((num_atributos, num_atributos), np.nan)
        attribute_names = [attr.nombre for attr in self.atributos]
        for i,attr1 in enumerate(self.atributos):
            [numeric_indices.append(i) if attr1.tipo== "numeric" else None]
            [categorical_indices.append(i) if attr1.tipo in ["character", "logical"] else None]
            for j, attr2 in enumerate(self.atributos):
                # Comprobamos el tipo de atributos
                if attr1.tipo == "numeric" and attr2.tipo == "numeric":
                    result[i,j] = ut.correlation_base(attr1.valores, attr2.valores)
                elif attr1.tipo in ["character", "logical"] and attr2.tipo in ["character", "logical"]:
                    result[i,j] = ut.mutualInformation_base(attr1.valores, attr2.valores)
                
        # Separamos la matriz en 2 matrices de correlaciones e información mutua
        correl_matrix = result[np.ix_(numeric_indices, numeric_indices)] if numeric_indices else None
        info_matrix = result[np.ix_(categorical_indices, categorical_indices)] if categorical_indices else None

        # Crear DataFrames con nombres de los atributos
        correl_df = pd.DataFrame(correl_matrix, 
                                 index=[attribute_names[i] for i in numeric_indices],
                                 columns=[attribute_names[i] for i in numeric_indices]) 
        info_df = pd.DataFrame(info_matrix, 
                               index=[attribute_names[i] for i in categorical_indices],
                               columns=[attribute_names[i] for i in categorical_indices])
    
        return {
            "correlations": correl_df,
            "mutual_information": info_df
        }
        
    def print(self):
        """Print the Dataset in a proper manner"""
        print("Dataset: ",self.nombre)
        for at in self.atributos:
            print("----------")
            at.print()

