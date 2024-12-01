import mlflow
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Função para realizar previsões futuras
def predict_future_with_saved_model(dataframe, column="Close", sequence_length=15, future_steps=7):
    """
    Carrega o modelo e realiza previsões futuras para os próximos `future_steps` dias.
    """
    # Carregar o modelo e o escalador do MLflow
    model = mlflow.keras.load_model("models:/lstm_model/latest")
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalizar os dados históricos
    scaled_data = scaler.fit_transform(dataframe[[column]])

    # Obter os últimos valores disponíveis
    last_sequence = scaled_data[-sequence_length:].flatten()
    predictions = []
    current_sequence = last_sequence.tolist()

    # Fazer previsões iterativamente
    for _ in range(future_steps):
        input_data = np.array(current_sequence[-sequence_length:]).reshape(1, sequence_length, 1)
        predicted_scaled = model.predict(input_data).flatten()[0]
        predictions.append(predicted_scaled)
        current_sequence.append(predicted_scaled)

    # Inverter a escala das previsões
    predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Gerar as datas futuras
    last_date = dataframe.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(future_steps)]

    return pd.DataFrame({"Date": future_dates, "Prediction": predictions_rescaled.flatten()}).set_index("Date")
