import time
import requests
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter, Gauge
from flask import Flask, request, jsonify

REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
REQUEST_LATENCY = Summary('inference_request_latency_seconds', 'Latency of inference requests in seconds')
INFERENCE_TIMESTAMP = Gauge('inference_last_timestamp', 'Timestamp of the last inference call')

app = Flask(__name__)

MLFLOW_MODEL_URI = "http://127.0.0.1:5001/invocations"  
@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time() 
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    INFERENCE_TIMESTAMP.set_to_current_time()

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input provided'}), 400

        input_df = pd.DataFrame([data])
        response = requests.post(MLFLOW_MODEL_URI, json={"dataframe_split": {
            "columns": input_df.columns.tolist(),
            "data": input_df.values.tolist()
        }})

        if response.status_code != 200:
            return jsonify({'error': 'Model server error', 'details': response.text}), 500

        result = response.json()
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        duration = time.time() - start_time
        print(f"[INFO] Inference took {duration:.4f} seconds.")

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter running on port 8000")

    app.run(port=5000)
