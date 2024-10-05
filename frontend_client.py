import socketio
import requests

sio = socketio.Client()

def connectToServer():
    sio.connect('http://localhost:5000', wait_timeout = 20)

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

@sio.on('frontend_hardware_output')
def data_processing(data):
    print(data)

@sio.on('frontend_anomaly_output')
def data_processing(data):
    print(data)

if __name__ == '__main__':
    timeframe = timeframe + ':01'
    timeframe = timeframe[6:10]+'-'+timeframe[3:5]+'-'+timeframe[0:2]+' '+timeframe[11:19]
    res = requests.post('https://7748-160-83-96-177.ngrok-free.app/load-prediction', json = {'timestamp':timeframe})
    print(res)
    print(res.json())     
    # connectToServer()