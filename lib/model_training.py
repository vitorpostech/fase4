import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.keras
from mlflow.exceptions import MlflowException

# Função para criar dados de entrada (X) e saída (y) para séries temporais
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Função para calcular métricas
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# Função principal para treinamento e registro no MLflow
def train_and_log_with_mlflow(dataframe_train, dataframe_test, log_preprocess, column="Close", sequence_length=15):
    """
    Treina o modelo e registra tudo no MLflow.
    """
    # Obter as datas mínimas e máximas para log
    train_min_date, train_max_date = dataframe_train.index.min(), dataframe_train.index.max()
    test_min_date, test_max_date = dataframe_test.index.min(), dataframe_test.index.max()

    # Normalização dos dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(dataframe_train[[column]])
    test_scaled = scaler.transform(dataframe_test[[column]])

    # Criar sequências para treino e teste
    X_train, y_train = create_sequences(train_scaled.flatten(), sequence_length)
    X_test, y_test = create_sequences(test_scaled.flatten(), sequence_length)

    # Ajustar forma dos dados para entrada no LSTM (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Construir o modelo LSTM
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(100, activation='relu'),
        Dense(1)
    ])

    # Compilar o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Iniciar o tracking do MLflow
    
    mlflow.start_run()
    try:
        # Treinar o modelo
        model.fit(X_train, y_train, epochs=100, batch_size=7, validation_data=(X_test, y_test), verbose=1)

        # Fazer previsões no conjunto de teste para avaliação
        predictions = model.predict(X_test)

        # Inverter a escala das previsões e valores reais para avaliação
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_rescaled = scaler.inverse_transform(predictions)

        # Avaliar o modelo
        metrics = evaluate_model(y_test_rescaled.flatten(), predictions_rescaled.flatten())

        # Logar métricas no MLflow
        mlflow.log_metric("MAE", metrics["MAE"])
        mlflow.log_metric("RMSE", metrics["RMSE"])
        mlflow.log_metric("MAPE", metrics["MAPE"])

        # Logar parâmetros do treinamento
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("train_min_date", train_min_date)
        mlflow.log_param("train_max_date", train_max_date)
        mlflow.log_param("test_min_date", test_min_date)
        mlflow.log_param("test_max_date", test_max_date)
        mlflow.log_param("log_preprocess", log_preprocess)

        # Logar o modelo como artefato
        mlflow.keras.log_model(model, artifact_path="lstm_model")

        # Registrar o modelo no Model Registry
        try:
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/lstm_model", "lstm_model")
        except MlflowException:
            print("O modelo já está registrado no MLflow Model Registry.")
    finally:
        mlflow.end_run()

    print("Treinamento e log concluídos!")