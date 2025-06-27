from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import random
import datetime

REQUEST_COUNT = Counter(
    'model_requests_total', 'Total jumlah permintaan ke model'
)

REQUEST_LATENCY = Summary(
    'model_request_latency_seconds', 'Waktu proses permintaan model'
)

LAST_INFERENCE_TIME = Gauge(
    'model_last_inference_timestamp', 'Waktu epoch terakhir kali prediksi dilakukan'
)

@REQUEST_LATENCY.time()
def inference_simulasi():
    REQUEST_COUNT.inc()
    LAST_INFERENCE_TIME.set_to_current_time()

    time.sleep(random.uniform(0.1, 0.5))

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        inference_simulasi()
        time.sleep(2)
