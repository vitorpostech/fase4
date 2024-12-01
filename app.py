import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify
from datetime import datetime
from dateutil.relativedelta import relativedelta
from lib import data_loader, model_training, model_forecast
from lib.preprocessing import TimeSeriesVerifier

app = Flask(__name__)

start= (datetime.now()- relativedelta(years=2)).strftime('%Y-%m-%d')
end= datetime.now().strftime('%Y-%m-%d')
symbol= 'MSFT'

# Endpoint para listar todos os usuários
@app.route('/train', methods=['GET'])
def train(symbol= symbol, start= start, end= end):
    data_loader.save_raw_data(symbol= symbol,
                            start_date= start, 
                            end_date= end)
    dataframe= data_loader.load_raw_data(f'data/raw/MSFT_{start}_{end}.pkl')
    dataframe_train= dataframe[:-30]
    dataframe_test= dataframe[-30:]

    verifier= TimeSeriesVerifier(dataframe_train, 'Close')

    dataframe_train, log= verifier.apply_verifications(['missing_data', 
                                                        'outliers_arima', 
                                                        'seasonality_trend', 
                                                        'distribution'])
    #print("\nLog de Verificações:")
    #print("\n".join(log))

    # Treinar e logar o modelo
    model_training.train_and_log_with_mlflow(dataframe_train, dataframe_test, column="Close", sequence_length=15, log_preprocess="\n".join(log))

    return {'response':'Model Trained Successfully'}

@app.route('/predict', methods=['GET'])
def predict(symbol= symbol, start= start, end= end):
    dataframe= data_loader.data_get(symbol= symbol, start_date= start, end_date= end)
    # Prever os próximos 7 dias
    future_predictions = model_forecast.predict_future_with_saved_model(dataframe, column="Close", sequence_length=15, future_steps=7)

    #print("\nPrevisões Finais para os Próximos 7 Dias:")
    #print(future_predictions)
    return jsonify(str(future_predictions))

#predict()
if __name__ == '__main__':
    app.run(debug=True)