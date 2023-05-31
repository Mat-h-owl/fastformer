#@title Librerias y Clase: DataFrameTransformer -- Version 2.1
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz, trapz
from scipy.signal import convolve, gaussian, savgol_filter
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt



class DataFrameTransformer:
    def __init__(self, df):
        self.df = df
        self.lasso = None

    def aplicar_filtro_gaussiano(self, ventana_gaussiana=11, sigma=1):
        def aplicar_filtro(df, filtro_gaussiano):
            columnas_filtradas = []
            for columna in df.columns:
                columna_filtrada = convolve(df[columna], filtro_gaussiano, mode='same') / sum(filtro_gaussiano)
                columnas_filtradas.append(columna_filtrada)
            
            df_filtrado = pd.DataFrame(np.column_stack(columnas_filtradas), columns=df.columns)
            return df_filtrado

        filtro_gaussiano = gaussian(ventana_gaussiana, sigma)
        self.df = aplicar_filtro(self.df, filtro_gaussiano)

    def derivada(self, columnas, ventana_savgol=11, orden_savgol=3):
        time = np.linspace(0, 100, len(self.df))
        time_step = time[1] - time[0]

        for columna in columnas:
            datos = self.df[columna].values
            datos_filtrados = savgol_filter(datos, ventana_savgol, orden_savgol, deriv=1, delta=time_step)
            self.df[f'dot_{columna}'] = datos_filtrados

    def ventana_de_integracion(self, columnas, ventana=7):
        for columna in columnas:
            self.df[f'int_{columna}'] = self.df[columna].rolling(window=ventana, min_periods=1).mean()

    def area_acumulada_uniforme(self, columnas_y, num_puntos_interpolados=1000):
        x_inicio, x_fin = 0, 100
        x_original = np.linspace(x_inicio, x_fin, len(self.df))

        for y_col in columnas_y:
            y_original = self.df[y_col].values
            x_interpolado = np.linspace(x_inicio, x_fin, num_puntos_interpolados)
            y_interpolado = np.interp(x_interpolado, x_original, y_original)

            areas_acumuladas = cumtrapz(y_interpolado, x_interpolado, initial=0)
            areas_acumuladas_original = np.interp(x_original, x_interpolado, areas_acumuladas)
            self.df[f'area_acum_{y_col}'] = areas_acumuladas_original

    def entrenar_modelo(self, variable_dependiente, factor_data_total=1, factor_data_entrenamiento=0.8,
                        lista=None, incluir_constante=False, return_metrics=False):
        if lista is None:
            lista = []

        df = self.df[:int(factor_data_total * len(self.df))]
        x = list(df.columns)
        x.remove(variable_dependiente)

        for e in lista:
            try:
                x.remove(e)
            except:
                print('---')

        X = df[x]
        y = df[variable_dependiente]

        split_point = int(factor_data_entrenamiento * len(X))

        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]

        if incluir_constante:
            X_train = sm.add_constant(X_train)
            X_val = sm.add_constant(X_val)
        else:
            print("Sin Constante")

        model = sm.OLS(y_train, X_train)

        results = model.fit()
        y_val_pred = results.predict(X_val)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        if return_metrics:
            return mae, r2

        print("Error absoluto medio (MAE):", mae)
        print("R^2 score:", r2)
        print(results.summary())
        plt.figure(figsize=(19, 6))
        t = np.arange(len(y_val))
        plt.plot(t, y_val, label='Valores reales')
        plt.plot(t, y_val_pred, label='Predicciones')
        plt.xlabel('Índice de los datos de prueba')
        plt.ylabel('Valor')
        plt.title('Comparación entre valores reales y predicciones del modelo')
        plt.legend()
        plt.show()

        return results


    def entrenar_modelo_lasso(self, variable_dependiente, factor_data_total=1, factor_data_entrenamiento=0.8,
                              lista=None, alpha=0.005, return_metrics=False):
        if lista is None:
            lista = []

        df = self.df[:int(factor_data_total * len(self.df))]
        x = list(df.columns)
        x.remove(variable_dependiente)

        for e in lista:
            try:
                x.remove(e)
            except:
                print('-')

        X = df[x]
        self.columnas_lasso = X.columns
        y = df[variable_dependiente]

        split_point = int(factor_data_entrenamiento * len(X))

        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]

        self.lasso = Lasso(alpha=alpha)
        self.lasso.fit(X_train, y_train)

        y_val_pred = self.lasso.predict(X_val)

        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        if return_metrics:
            return mae, r2

        print("Error absoluto medio (MAE):", mae)
        print("R^2 score:", r2)
        plt.figure(figsize=(19, 6))
        t = np.arange(len(y_val))
        plt.plot(t, y_val, label='Valores reales')
        plt.plot(t, y_val_pred, label='Predicciones Lasso')
        plt.xlabel('Índice de los datos de prueba')
        plt.ylabel('Valor')
        plt.title('Comparación entre valores reales y predicciones del modelo Lasso')
        plt.legend()
        plt.show()


    def analizar_influencia_variables(self, variable_objetivo, columnas_eliminar=None, modelo='regular'):
        if columnas_eliminar is None:
            columnas_eliminar = []

        columnas = self.df.columns
        columnas = columnas.drop(variable_objetivo)
        columnas = columnas.drop(columnas_eliminar)

        resultados = []

        for col in columnas:
            print(f"Eliminando columna '{col}':")
            if modelo == 'lasso':
                mae, r2 = self.entrenar_modelo_lasso(variable_objetivo, lista=[col], return_metrics=True)
            elif modelo == 'regular':
                mae, r2 = self.entrenar_modelo(variable_objetivo, lista=[col], return_metrics=True)
            else:
                raise ValueError(f"El modelo '{modelo}' no es válido. Por favor, elija 'regular' o 'lasso'.")
            resultados.append({"columna": col, "MAE": mae, "R^2": r2})
            print("---------------------------------------------------")

        resultados_df = pd.DataFrame(resultados)
        return resultados_df
