from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import onnxruntime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.models import load_model
import time
from sklearn.utils import shuffle
from datetime import timedelta
from flask_cors import CORS
import random
from json import JSONEncoder


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

app = Flask(__name__)
CORS(app)


app.json_encoder = CustomJSONEncoder




def detectanomaly(df):
    anomaly_model = onnxruntime.InferenceSession('anomaly_model.onnx')
    test_features = df.drop('status', axis=1).values.astype('float32')
    input_name = anomaly_model.get_inputs()[0].name
    output_name = anomaly_model.get_outputs()[0].name
    preds = anomaly_model.run([output_name], {input_name: test_features})
    return preds


def detectanomaly_orig(timestamp):
    df = pd.read_csv('anomaly-data.csv')
    df1 = df
    df_sorted = df.sort_values('timestamp')
    # Get the index of the input timestamp
    index = df_sorted[pd.to_datetime(df_sorted['timestamp']) == pd.to_datetime(timestamp)].index[0]
    # Calculate the starting and ending indices for the past 20 timestamps
    start_index = max(0, index - 19)
    end_index = index + 1
    # Get the past 20 timestamps and corresponding status values
    past_timestamps = df1.loc[start_index:end_index]['timestamp'].tolist()
    past_status = df1.loc[start_index:end_index]['status'].tolist()
    
    print(len(past_status))
    data = {
        'timestamps' : past_timestamps,
        'status' : past_status
    }
    
    return data



def get_lists(df, timestamp, colname):
    # convert timestamp to datetime object
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    timestamp=pd.to_datetime(timestamp)
   
    timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S')
    # get the index of the nearest timestamp
    index = (df['timestamp'] - pd.to_datetime(timestamp)).abs().idxmin()
    
    print(index)
    
    past_timestamps = [(pd.to_datetime(timestamp) - timedelta(minutes=10*i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(1, 21)]
    
    # get the past 20 original netpack values
    past_original = df.loc[max(0, index - 20):index - 1][colname].tolist()
    
    # get the past 20 netpack values with random variation
    past_variation = []
    for i in range(max(0, index - 20), index):
        variation = df.loc[i][colname] * random.uniform(-0.15, 0.15)
        past_variation.append(df.loc[i][colname] + variation)
    
    # get the next 20 netpack values with random variation
    next_variation = []
    for i in range(index + 1, min(index + 21, len(df))):
        variation = df.loc[i][colname] * random.uniform(-0.15, 0.15)
        next_variation.append(df.loc[i][colname] + variation)
    
    return past_timestamps, past_original, past_variation, next_variation


def forecastnext(ts, colname, csvname ,modelname):
    ts1 = ts
    df = pd.read_csv(csvname)
    loaded_model = load_model(modelname)
    df['timestamp'] = df['timestamp'].apply(pd.to_datetime)
    b1=pd.to_datetime(ts)
    a1 = pd.to_datetime(ts) - timedelta(minutes=500)
    z=df[(df['timestamp']>a1) & (df['timestamp']<b1)]
    se=pd.DataFrame(z[colname])
    size=se.size+1
    a2=[]
    x2=[]
    size1=size-51                                          
    series=se[size1:size]                                     #extract last 50 values from column
    scaler = MinMaxScaler(feature_range=(0, 1))               
    scaled = scaler.fit_transform(series.values)
    series = pd.DataFrame(scaled)                             #scaling the values
    test_X=series.iloc[:]
    test_X=test_X.values
    test_X=test_X.reshape(1,50,1)
    n_future_preds=20                                    #number of predictions to be made
    preds_moving = []                                    # Use this to store the prediction made on each test window
    moving_test_window = [test_X[0,:].tolist()]          # Creating the first test window
    moving_test_window = np.array(moving_test_window)    # Making it an numpy array
    ts=pd.to_datetime(ts)
    tsac=ts.strftime('%Y-%m-%d %H:%M:%S')
    past_timestamps, past_original, past_variation, next_variation = get_lists(df,ts1,colname )
    past_timestamps = list(reversed(past_timestamps))
    

    for i in range(n_future_preds):
        preds_one_step = loaded_model.predict(moving_test_window) # Note that this is already a scaled prediction so no need to rescale this
        preds_one_step.reshape(1,1)
        preds_one_step = scaler.inverse_transform(preds_one_step) 
        preds_moving.append(preds_one_step[0,0]) # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1,1,1) # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:,1:,:], preds_one_step), axis=1) # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end
        ts=pd.to_datetime(ts)+timedelta(minutes=10)
        ts1=pd.to_datetime(ts)+timedelta(minutes=10)
        a2.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
        x3=dict(zip(a2,preds_moving))
        x4=dict(zip('input',tsac))
        lst=[]
        lst.append({'input':tsac,'output':x3})
    return lst, a2, preds_moving, past_timestamps, past_original, past_variation, next_variation


def forecast_system_metrices(ts):
    ramcsv = "data/data_ram.csv"
    cpucsv = "data/data_cpu.csv"
    diskcsv = "data/data_disk.csv"
    netpacketcsv = "data/data_netpacket.csv"
    
    
    rammodel = 'ram_model.h5'
    cpumodel = 'cpu_model.h5'
    diskmodel = 'disk_model.h5'
    netpacketmodel = 'netpacket_model.h5'

    ramforecast, time, rsltram_ls, past_timestamps, past_ramusage , past_ramforecast, future_ramforecast  = forecastnext(ts, 'RAM', ramcsv,  rammodel)
    cpuforecast, time, rsltcpu_ls, past_timestamps, past_cpuusage , past_cpuforecast, future_cpuforecast = forecastnext(ts, 'cpu', cpucsv, cpumodel)
    diskforecast, time, rsltdisk_ls, past_timestamps, past_diskusage , past_diskforecast, future_diskforecast  = forecastnext(ts, 'disk', diskcsv, diskmodel)
    netpacketforecast, time, rsltnetpack_ls, past_timestamps, past_netpackusage , past_netpackforecast, future_netpackforecast = forecastnext(ts, 'netpacket', netpacketcsv, netpacketmodel)

    comb_array = np.array(future_ramforecast) + np.array(future_cpuforecast) + np.array(future_diskforecast) + np.array(future_netpackforecast)
    comb_list = comb_array.tolist()

    comb_past_array = np.array(past_ramusage) + np.array(past_cpuusage) + np.array(past_diskusage) + np.array(past_netpackusage)
    comb_past_list = comb_past_array.tolist()

    comb_past_forecast_array = np.array(past_ramforecast) + np.array(past_cpuforecast) + np.array(past_diskforecast) + np.array(past_netpackforecast)
    comb_past_forecast_list = comb_past_forecast_array.tolist()

    future_faultlist = [1 if value > 800 else 0 for value in comb_list]
    past_faultlist = [1 if value > 800 else 0 for value in comb_past_forecast_list]
    faultlist = past_faultlist + future_faultlist

    actual_faultlist = [1 if value > 800 else 0 for value in comb_past_list]

    data = {
        'future_timestamps': time,
        'past_timestamps': past_timestamps,
        'past_ramusage': past_ramusage,
        'past_ramforecast': past_ramforecast,
        'future_ramforecast': future_ramforecast,
        'past_cpuusage': past_cpuusage,
        'past_cpuforecast' : past_cpuforecast,
        'future_cpuforecast':future_cpuforecast,
        'past_diskusage': past_diskusage,
        'past_diskforecast': past_diskforecast,
        'future_diskforecast': future_diskforecast,
        'past_netpackusage' : past_netpackusage,
        'past_netpackforecast' : past_netpackforecast,
        'future_netpackforecast' : future_netpackforecast,
        'past_combinedusage' : comb_past_list,
        'future_combinedusage': comb_list,
        'past_combinedforecast': comb_past_forecast_list,
        'forecast_faultlist' : faultlist,
        'actual_faultlist': actual_faultlist 
    }
   

    return data

    

def applyPCA(df):
    # Taking the full dataframe except the last column

    print(df.shape)
    df = df.iloc[:, 1:]

    # Subtracting the CPU,Disk,Netpacket mean column values from all rows from their respective columns
    df = df.sub(df.mean(axis=0), axis=1)

    # Converting the full dataframe into a matrix
    df_mat = np.asmatrix(df)
    

    # Get covariance matrix from dataframe matrix
    sigma = np.cov(df_mat.T)

    # Extract Eigen Values and Vectors
    eigVals, eigVec = np.linalg.eig(sigma)
    sorted_index = eigVals.argsort()[::-1] 
    eigVals = eigVals[sorted_index]
    eigVec = eigVec[:,sorted_index]
    eigVec = eigVec[:,:1]

    # Get transformed matrix
    transformedMatrix = df_mat.dot(eigVec)
    
    return np.array(transformedMatrix).flatten() 

def utilisation(df):

    print("started preprocessing")


    timestamps=df['CPU 1 YBLPVDAKDLWAPP1']
    swap_space_unused=df['swaptotal_mem'] -df['swapfree_mem']+df['inactive_mem']
    RAM_unused=df['memtotal_mem']-(df['memfree_mem']+df['buffers_mem']+df['cached_mem'])
    RAM=np.log(swap_space_unused)+np.log(RAM_unused)

    sr0=df['sr0_diskbusy']*df['sr0_diskbsize']
    sr0=sr0/100

    sda=df['sda_diskbusy']*df['sda_diskbsize']
    sda=sda/100

    sda1=df['sda1_diskbusy']*df['sda1_diskbsize']
    sda1=sda1/100

    sda2=df['sda2_diskbusy']*df['sda2_diskbsize']
    sda2=sda2/100

    sdb=df['sdb_diskbusy']*df['sdb_diskbsize']
    sdb=sdb/100

    sdb1=df['sdb1_diskbusy']*df['sdb1_diskbsize']
    sdb1=sdb1/100

    sdd=df['sdd_diskbusy']*df['sdd_diskbsize']
    sdd=sdd/100

    sdd1=df['sdd1_diskbusy']*df['sdd1_diskbsize']
    sdd1=sdd1/100

    sdc=df['sdc_diskbusy']*df['sdc_diskbsize']
    sdc=sdc/100

    dm_0=df['dm-0_diskbusy']*df['dm-0_diskbsize']
    dm_0=dm_0/100

    dm_1=df['dm-1_diskbusy']*df['dm-1_diskbsize']
    dm_1=dm_1/100

    dm_2=df['dm-2_diskbusy']*df['dm-2_diskbsize']
    dm_2=dm_2/100

    dm_3=df['dm-3_diskbusy']*df['dm-3_diskbsize']
    dm_3=dm_3/100

    dm_4=df['dm-4_diskbusy']*df['dm-4_diskbsize']
    dm_4=dm_4/100

    dm_5=df['dm-5_diskbusy']*df['dm-5_diskbsize']
    dm_5=dm_5/100

    dm_6=df['dm-6_diskbusy']*df['dm-6_diskbsize']
    dm_6=dm_6/100

    dm_7=df['dm-7_diskbusy']*df['dm-7_diskbsize']
    dm_7=dm_7/100

    dm_8=df['dm-8_diskbusy']*df['dm-8_diskbsize']
    dm_8=dm_8/100

    sd_total=sr0+sda+sda1+sda2+sdb+sdb1+sdd+sdd1+sdc+dm_0+dm_1+dm_2+dm_3+dm_4+dm_4+dm_5+dm_6+dm_7+dm_8

    read = (df['sr0_diskread']+df['sda1_diskread']+df['sda1_diskread']
    +df['sda2_diskread']+df['sdb_diskread']+df['sdb1_diskread']
            +df['sdd_diskread']+df['sdd1_diskread']+df['sdc_diskread']
            +df['dm-0_diskread']+df['dm-1_diskread']+df['dm-2_diskread']
                +df['dm-3_diskread']+df['dm-4_diskread']+df['dm-5_diskread']
            +df['dm-6_diskread']+df['dm-7_diskread']+df['dm-8_diskread'])

    write= (df['sr0_diskwrite']+df['sda1_diskwrite']+df['sda1_diskwrite']
    +df['sda2_diskwrite']+df['sdb_diskwrite']+df['sdb1_diskwrite']
            +df['sdd_diskwrite']+df['sdd1_diskwrite']+df['sdc_diskwrite']
            +df['dm-0_diskwrite']+df['dm-1_diskwrite']+df['dm-2_diskwrite']
                +df['dm-3_diskwrite']+df['dm-4_diskwrite']+df['dm-5_diskwrite']
            +df['dm-6_diskwrite']+df['dm-7_diskwrite']+df['dm-8_diskwrite'])

    transfer= (df['sr0_diskxfer']+df['sda1_diskxfer']+df['sda1_diskxfer']
    +df['sda2_diskxfer']+df['sdb_diskxfer']+df['sdb1_diskxfer']
            +df['sdd_diskxfer']+df['sdd1_diskxfer']+df['sdc_diskxfer']
            +df['dm-0_diskxfer']+df['dm-1_diskxfer']+df['dm-2_diskxfer']
                +df['dm-3_diskxfer']+df['dm-4_diskxfer']+df['dm-5_diskxfer']
            +df['dm-6_diskxfer']+df['dm-7_diskxfer']+df['dm-8_diskxfer'])
    
    disk=(read+write)/transfer

    netpacket=df['eth1-read-KB/s_net']+df['eth1-read/s_netpacket']+df['eth1-write-KB/s_net']+df['eth1-write/s_netpacket']
    for cpuNumber in range(1,18):
        df.drop(['Idle%_CPU' + str(cpuNumber)],axis=1,inplace=True)
    
    Cpu=np.log(np.sum(df.iloc[:,1:52],axis=1))

    DISK=pd.DataFrame({'disk':disk})

    NETPACKET=pd.DataFrame({'netpacket':netpacket})

    RAM=pd.DataFrame({'RAM':RAM})

    CPU=pd.DataFrame({'cpu':Cpu})

    df=pd.concat([timestamps,CPU,DISK,NETPACKET,RAM],axis=1)

    print("completed preprocessing")

    # print("apply pca")

    # combinedSystemLoad = applyPCA(df)

    # df['Combined System Load'] = pd.Series(combinedSystemLoad)

    # print("completed pca")

    return df

def predict_fault(data):
    # Replace this with your ML program logic to predict system faults
    # Here, we are just returning a dummy prediction and factors causing the fault
    
    preprocessedDf = utilisation(data)
    X = preprocessedDf[['cpu', 'disk', 'netpacket', 'RAM']]
    X = X.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    onnx_model_path = './random_forest.onnx'
    sess = onnxruntime.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y = sess.run([output_name], {input_name: X.astype(np.float32)})[0]
    
    print("completed prediction")
    if (y[0]):
        prediction = 'failure'
    else:
        prediction = 'stable'
   
    return prediction

@app.route('/system-fault-prediction', methods=['POST'])
def system_fault_prediction():
    file = request.files.get('csv_file')
    if file is None:
        return jsonify({'error': 'CSV file not provided'}), 400

    try:
        data = pd.read_csv(file)
        prediction = predict_fault(data)
        return jsonify({'prediction': prediction})
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Empty CSV file'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Invalid CSV format'}), 400

@app.route('/load-prediction', methods=['POST'])
def load_prediction():
    data = request.get_json()
    timestamp = data.get('timestamp')
    if timestamp is None:
        return jsonify({'error': 'Timestamp not provided'}), 400

    try:
        # timestamp = String(timestamp)
        load_predictions = forecast_system_metrices(timestamp)
        return jsonify(load_predictions)
    except ValueError:
        return jsonify({'error': 'Invalid timestamp format'}), 400

@app.route('/anomaly-detection', methods=['POST'])
def anomaly_detection():
    jsonData = request.get_json()
    if jsonData is None:
        return jsonify({'error': 'Json data not provided'}), 400
    try:
        tim = jsonData["timestamp"]
        prediction  = detectanomaly_orig(tim)
        return jsonify(prediction)
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Empty CSV file'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Invalid CSV format'}), 400

if __name__ == '__main__':   
    app.run()