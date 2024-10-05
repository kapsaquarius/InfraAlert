import socketio
import requests
from datetime import datetime

sio = socketio.Client()

def connectToServer():
    # sio.connect('https://3aa0-160-83-96-177.ngrok.io', wait_timeout = 20)
    sio.connect('https://hackathon-backend-flask-1-d2iqcgwvfa-uc.a.run.app', wait_timeout = 20)

@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")
    print('Exiting the program!!!')
    exit(0)

@sio.on('model_input')
def data_processing(data):
    print('Reached Here!!!')
    if(len(data)!=0):
        print(data['data']['hardwareData'])
        hardware_stats = get_hardware_stats(data['data']['hardwareData'].split(',')[0])
        network_anomaly_data = get_network_anomaly_data(data['data']['networkData'])
        sio.emit('model_hardware_output', hardware_stats)
        sio.emit('network_anomaly_output', network_anomaly_data)


def get_hardware_stats(timeframe):
    
    timeframe = timeframe + ':01'
    timeframe = timeframe[6:10]+'-'+timeframe[3:5]+'-'+timeframe[0:2]+' '+timeframe[11:19]
    # res = requests.post('https://7748-160-83-96-177.ngrok-free.app/load-prediction', json = {'timestamp':timeframe})
    res = requests.post('https://hackathon-ml-server-d2iqcgwvfa-uc.a.run.app/load-prediction', json = {'timestamp':timeframe})
    return res.json()


def get_network_anomaly_data(data):
    # res = requests.post('https://7748-160-83-96-177.ngrok-free.app/anomaly-detection', json = data)
    res = requests.post('https://hackathon-ml-server-d2iqcgwvfa-uc.a.run.app/anomaly-detection', json = data)
    return res.json()

if __name__ == '__main__':
    connectToServer()