import socketio
import csv
import time

sio = socketio.Client()

@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error():
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")
    print('Exiting the program!!!')
    exit(0)


@sio.on('model_input')
def test_fun(data):
    print('Reached Here!!!')

def connectToServer():
    sio.connect('https://hackathon-backend-flask-1-d2iqcgwvfa-uc.a.run.app', wait_timeout = 20)
    start_data_streaming()

def start_data_streaming():
    count = 0
    with open('data_source/final_combined_system_metric_data.csv', newline='') as hardwareCsv , open('data_source/anomaly-data.csv', newline='') as networkCsv:
        hardwareData = csv.reader(hardwareCsv, delimiter=',')
        networkData = csv.reader(networkCsv, delimiter=',')
        print("Something happened!!!")
        for (hardware, network) in zip(hardwareData, networkData):
            if(count==0):
                count+=1
                continue
            count+=1
            time.sleep(20)
            network_ex_data = {"timestamp":"06-01-2023 07:51:00", "duration":0,"src_bytes":226,"dst_bytes":2973,"land":0,"wrong_fragment":0,"urgent":0,"hot":0,"num_failed_logins":0,"logged_in":1,"num_compromised":0,"root_shell":0,"su_attempted":0,"num_root":0,"num_file_creations":0,"num_shells":0,"num_access_files":0,"num_outbound_cmds":0,"is_host_login":0,"is_guest_login":0,"count":14,"srv_count":15,"serror_rate":0,"srv_serror_rate":0,"rerror_rate":0,"srv_rerror_rate":0,"same_srv_rate":1,"diff_srv_rate":0,"srv_diff_host_rate":0.13,"dst_host_count":58,"dst_host_srv_count":255,"dst_host_same_srv_rate":1,"dst_host_diff_srv_rate":0,"dst_host_same_src_port_rate":0.02,"dst_host_srv_diff_host_rate":0.01,"Unnamed: 34":0,"Unnamed: 35":0,"Unnamed: 36":0,"dst_host_srv_rerror_rate":0,"status":"normal"}
            print({'hardwareData': 'Timestamp: ' + ", ".join(hardware),
                                        'networkData': ", ".join(network),
                                        'count': count})
            networkData = {}
            for (key , val) in zip(network_ex_data.keys(), network):
                networkData[key]=val
            print(networkData)
            sio.emit('data_source', {'hardwareData': ",".join(hardware),
                                        'networkData': networkData,
                                        'count': count})

if __name__ == '__main__':
    connectToServer()