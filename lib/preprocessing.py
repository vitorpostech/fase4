import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

class TimeSeriesVerifier:
    def __init__(self, dataframe, column):
        """
        Inicializa a classe com o DataFrame e a coluna de série temporal.
        
        :param dataframe: DataFrame contendo a série temporal.
        :param column: Nome da coluna com os dados da série temporal.
        """
        self.dataframe = dataframe.copy()
        self.column = column
        self.result_log = []

    def verify_missing_data(self):
        # Contar valores ausentes
        null_count = self.dataframe[self.column].isnull().sum()
        if null_count > 0:
            self.result_log.append(f"Valores ausentes encontrados: {null_count}.")
            # Preencher com interpolação linear
            self.dataframe[self.column] = self.dataframe[self.column].interpolate(method='linear', limit_direction='both')
            # Verificar novamente se há valores ausentes
            remaining_nulls = self.dataframe[self.column].isnull().sum()
            if remaining_nulls > 0:
                self.result_log.append(f"Após interpolação, valores ausentes restantes: {remaining_nulls}. Preenchendo com forward-fill.")
                self.dataframe[self.column] = self.dataframe[self.column].fillna(method='ffill')
            self.result_log.append("Valores ausentes preenchidos com interpolação linear e forward-fill.")

    def verify_zero_values(self):
        # Contar valores iguais a zero
        zero_count = (self.dataframe[self.column] == 0).sum()
        if zero_count > 0:
            self.result_log.append(f"Valores zerados encontrados: {zero_count}.")
            # Substituir valores zero por interpolação
            self.dataframe[self.column] = self.dataframe[self.column].replace(0, np.nan).interpolate(method='linear', limit_direction='both')
            self.result_log.append("Valores zero corrigidos usando interpolação linear.")

    def verify_outliers_arima(self):
        try:
            model = ARIMA(self.dataframe[self.column].fillna(method='bfill'), order=(1, 1, 1))
            fitted_model = model.fit()
            predicted = fitted_model.predict()
            residuals = self.dataframe[self.column] - predicted
            threshold = 3 * np.std(residuals)
            outliers = residuals.abs() > threshold
            outlier_count = outliers.sum()
            if outlier_count > 0:
                self.dataframe.loc[outliers, self.column] = predicted[outliers]
                self.result_log.append(f"Outliers corrigidos: {outlier_count} valores ajustados com ARIMA.")
        except Exception as e:
            self.result_log.append(f"Erro ao corrigir outliers com ARIMA: {e}")

    def verify_stationarity(self):
        def adf_test(series):
            result = adfuller(series.dropna())
            return result[1] <= 0.05

        def kpss_test(series):
            result = kpss(series.dropna(), regression="c")
            return result[1] > 0.05

        stationary_adf = adf_test(self.dataframe[self.column])
        stationary_kpss = kpss_test(self.dataframe[self.column])

        if not (stationary_adf and stationary_kpss):
            self.dataframe[self.column] = np.log(self.dataframe[self.column].clip(lower=1))  # Log-transform
            self.dataframe[self.column] = self.dataframe[self.column].diff().dropna()  # First differencing
            self.result_log.append("Série transformada para estacionariedade (log + diferenciação).")
        else:
            self.result_log.append("Série já estacionária.")

    def verify_seasonality_trend(self):
        try:
            filled_series = self.dataframe[self.column].fillna(method='bfill')
            decomposition = seasonal_decompose(filled_series, model='additive', period=12)
            if decomposition.seasonal.any() or decomposition.trend.any():
                self.result_log.append("Sazonalidade e/ou tendência identificadas na série.")
        except Exception as e:
            self.result_log.append(f"Erro ao decompor série temporal: {e}")

    def apply_verifications(self, checks):
        """
        Aplica as verificações e correções solicitadas.
        
        :param checks: Lista de verificações a serem aplicadas.
        """
        if "missing_data" in checks:
            self.verify_missing_data()
        if "zero_values" in checks:
            self.verify_zero_values()
        if "outliers_arima" in checks:
            self.verify_outliers_arima()
        if "stationarity" in checks:
            self.verify_stationarity()
        if "seasonality_trend" in checks:
            self.verify_seasonality_trend()

        return self.dataframe, self.result_log