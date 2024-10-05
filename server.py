from threading import Lock
from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit

async_mode = None

app = Flask(__name__)
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins='*')
thread = None
thread_lock = Lock()

@socketio.on('data_source')
def metrics_pipeline(data):
    print(data)
    emit('model_input',
         {'data': data}, broadcast=True)
    
@socketio.on('model_hardware_output')
def metrics_output(data):
    emit('frontend_hardware_output', data, broadcast=True)

@socketio.on('network_anomaly_output')
def metrics_output(data):
    emit('frontend_anomaly_output', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app)
            