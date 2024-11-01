# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:27:17 2024

@author: Iosu Abal
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc(dataset, numeric_attr):
    """
    Method to plot the area under curve based on the numeric attribute parameter.
    
    Parameters:
        dataset: The DataSet containing both numerical values and logical values. Otherwise, an error is thrown.
        numeric_attr: The name of the numeric attribute to use for plotting.
    Returns:
            None: This function does not return a value. It displays the plot directly."""
    ratesDataset = dataset.calculate_rates(numeric_attr)  # Nuestra funcion de Dataset

    # Convertir a df para poder plottear
    rates_df = pd.DataFrame({'TPR': ratesDataset.atributos[0].valores, 'FPR': ratesDataset.atributos[1].valores})
    auc = dataset.calculate_metrics()[numeric_attr]['AUC']
    plt.figure(figsize=(10, 6))
    # Sombrear el area
    plt.fill_between(rates_df['FPR'], rates_df['TPR'], color='blue', alpha=0.8)
    plt.plot([0, 1], [0, 1], color='red', linewidth=0.7, linestyle='--', label='Clasificador aleatorio')
    plt.title('Area Under Curve')
    plt.text(0.85, 0.05, f'AUC = {auc:.3f}', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('FPR (False Positive Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def plot_correlation(dataset):
        """Create two heatmaps to visualise the correlation and mutual information of a Dataset. 
        This method calls the 'correlation' function defined in the Dataset class
        Parameters:
            dataset: The dataset containing both numerical and discrete atributes. If it doesn't have one of those, only one 
                     heatmap will be created.
        Returns:
            None: This function does not return a value. It displays heatmaps directly."""
        
        # Llamamos a la funcion correlation de la clase Dataset
        matrices = dataset.correlation()
        dfCorrelacion = matrices['correlations']
        dfInformacion = matrices['mutual_information']
        # Una figura de 12x6 (mas filas que columnas)
        plt.figure(figsize=(12, 6))

        if not dfCorrelacion.empty:
            plt.subplot(1, 2, 1)
            heatmap_cor = sns.heatmap(dfCorrelacion,
                                       cmap='coolwarm',
                                       vmin=-1, vmax=1, #la correlacion siempre estara entre 0 y 1
                                       cbar_kws={"label": "Correlación"},
                                       annot=True,
                                       fmt=".2f") #redondeado a dos decimales
            plt.title('Heatmap de Correlación')
            plt.xlabel('Columnas')
            plt.ylabel('Filas')

        #Aplicamos lo mismo para la informacion mutua
        if not dfInformacion.empty:
            plt.subplot(1, 2, 2)
            heatmap_inf = sns.heatmap(dfInformacion,
                                       cmap='YlOrRd',
                                       vmin=0, vmax=1,
                                       cbar_kws={"label": "Información Mutua"},
                                       annot=True,
                                       fmt=".2f")
            plt.title('Heatmap de Información Mutua')
            plt.xlabel('Columnas')
            plt.ylabel('Filas')
        
        plt.show()