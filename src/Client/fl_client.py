import pickle
import numpy as np
import pandas as pd
import random
import time
import os
import argparse
import codecs

from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import socket

ipAddress = ""

class LocalModel(object):
    def __init__(self, model_config, data_collected):
        # model_config:
        # 'model': self.global_model.model.to_json(),
        # 'model_id'
        # 'min_train_size'
        # 'data_split': (0.6, 0.3, 0.1), # train, test, valid
        # 'epoch_per_round'
        # 'batch_size'
        self.model_config = model_config

        self.model = model_from_json(model_config['model_json'])
        # the weights will be initialized on first pull from server

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("compile done", self.model.summary())

        train_data, test_data, valid_data = data_collected
        self.x_train = train_data[0]
        self.y_train = train_data[1]
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.x_valid = valid_data[0]
        self.y_valid = valid_data[1]

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def  train_one_round(self):
        print('Batch: ', self.model_config['batch_size'])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit(x=self.x_train, y=self.y_train,
                       epochs=self.model_config['epoch_per_round'],
                       batch_size=self.model_config['batch_size'],
                       # verbose=1,
                       validation_data=(self.x_valid, self.y_valid))

        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        return self.model.get_weights(), score[0], score[1]

    def validate(self):
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        print('Validate loss:', score[0])
        print('Validate accuracy:', score[1])
        return score

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score


# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.


def train_test_valid_set(kdd_temp, kdd, kdd_t):
    # kdd_full_cols holding all the values for the categorical features, total size: (122 + class)
    kdd_full_cols = [kdd_temp.columns[0]] + sorted(list(set(kdd_temp.protocol_type.values))) + sorted(
        list(set(kdd_temp.service.values))) + sorted(list(set(kdd_temp.flag.values))) + kdd_temp.columns[4:].tolist()

    # for the cat features (in train and test sets): remove its original column and generate its dummy values
    cat_lst = ['protocol_type', 'service', 'flag']
    for col in cat_lst:
        kdd = cat_encode(kdd, col)
        kdd_t = cat_encode(kdd_t, col)
    # print(kdd['duration'].describe())

    # su_attempted column has a max value of 2.0, and is supposed to be binary feature.
    # Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0
    kdd['su_attempted'].replace(2, 0, inplace=True)
    kdd_t['su_attempted'].replace(2, 0, inplace=True)

    # Next, we notice that the num_outbound_cmds column only takes on one value!
    # Now, that's not a very useful feature - let's drop it from the data set
    kdd.drop('num_outbound_cmds', axis=1, inplace=True)
    kdd_t.drop('num_outbound_cmds', axis=1, inplace=True)
    kdd_full_cols.remove('num_outbound_cmds')

    '''
    # *** Feature Reduction ***
    features_to_remove = ['land', 'wrong_fragment', 'urgent', 'num_failed_logins', 'root_shell', 'su_attempted',
                          'num_file_creations', 'num_access_files', 'is_host_login', 'dst_host_count',
                          'dst_host_rerror_rate']

    kdd = kdd.drop(features_to_remove, axis=1)
    kdd_t = kdd_t.drop(features_to_remove, axis=1)

    for key in range(len(features_to_remove)):
        kdd_full_cols.remove(features_to_remove[key])
    '''

    # np.log1p is used to correct the skew in the columns
    log_lst = ['duration', 'src_bytes', 'dst_bytes']
    for col in log_lst:
        kdd[col] = log_trns(kdd, col)
        kdd_t[col] = log_trns(kdd_t, col)
    # print(kdd['duration'].describe())

    for col in kdd_full_cols:
        if col not in kdd.columns:
            kdd[col] = 0

    for col in kdd_full_cols:
        if col not in kdd_t.columns:
            kdd_t[col] = 0

    kdd.columns = map(str.lower, kdd.columns)
    kdd = kdd.reindex(sorted(kdd.columns), axis=1)

    kdd_t.columns = map(str.lower, kdd_t.columns)
    kdd_t = kdd_t.reindex(sorted(kdd_t.columns), axis=1)

    # Remove the 'class' column from train/test sets
    train_y = kdd.pop('class')
    y_test = kdd_t.pop('class')

    # Generate dummy values for the label columns in train/test sets
    train_y = pd.get_dummies(train_y)
    y_test = pd.get_dummies(y_test)

    # target = train_y; train = train_x; test = test_x; y_test = test_y
    train_x = kdd.values
    train_y = train_y.values
    test_x = kdd_t.values
    test_y = y_test.values

    # We rescale features to [0, 1]
    min_max_scaler = MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    test_x = min_max_scaler.transform(test_x)

    print("Nb of input: ", train_x.shape[1])

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=1 / 10, random_state=1)

    return (train_x, train_y), (test_x, test_y), (val_x, val_y)


def load_data_set(train_file_name, test_file_name):
    #os.chdir('../')
    #os.chdir("..")
    #os.chdir("..")

    with open('Datasets/kddcup.names.txt',
              'r') as infile:
        kdd_names = infile.readlines()
    print("kddcup done")
    # kdd_cols holding the 41 features in the data set
    kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]

    # The Train+/Test+ data sets include sample difficulty rating and the attack class
    kdd_cols += ['class']
    # print(kdd.head())

    train_file_name = train_file_name + '.csv'

    test_file_name = test_file_name + '.csv'


    kdd_temp = pd.read_csv('Datasets/KDDTrainForDummy.csv', names=kdd_cols)
    kdd = pd.read_csv(train_file_name, names=kdd_cols)
    kdd_t = pd.read_csv(test_file_name, names=kdd_cols)


    return kdd_temp, kdd, kdd_t


class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 1200


    def __init__(self, client_file):
        self.local_model = None
        self.client_file = client_file
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        print(IPAddr)

        self.sio = SocketIO(ipAddress, 5000, LoggingNamespace)
        self.register_handles()
        print("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    # ** Socket Event Handler **
    def on_init(self, *args):
        model_config = args[0]
        print('on init', model_config)
        print('preparing local data based on server model_config')
        client_train_file = 'Datasets/ClientsDS/Unbalanced/Train/Client_' + str(self.client_file)
        client_test_file = 'Datasets/ClientsDS/Unbalanced/Test/Client_' + str(self.client_file)
        print('Train file', client_train_file)
        print('Test file', client_test_file)
        kdd_temp, kdd, kdd_t = load_data_set(client_train_file, client_test_file)

        data_distributed = train_test_valid_set(kdd_temp, kdd, kdd_t)

        self.local_model = LocalModel(model_config, data_distributed)

        with open('Datasets/ClientsDS/TimeZones/Client_' + str(self.client_file) + '.txt',
                  'r') as infile:
            client_time_zone = infile.read()

        # ready to be dispatched for training
        self.sio.emit('client_ready', {
            'train_size': self.local_model.x_train.shape[0],
            'client_time_zone': client_time_zone,
            'client_ID': str(self.client_file),
        })

    def register_handles(self):
        # *** Socket IO messaging ***
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            req = args[0]
            # req:
            #     'model_id'
            #     'round_number'
            #     'current_weights'
            #     'weights_format'
            #     'run_validation'
            print("update requested")

            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])

            self.local_model.set_weights(weights)
            print('\n______________\nTraining at round: ', req['round_number'], '\n______________')
            my_weights, train_loss, train_accuracy = self.local_model.train_one_round()
            resp = {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'valid_size': self.local_model.x_valid.shape[0],
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'client_ID': str(self.client_file),
            }
            if req['run_validation']:
                valid_loss, valid_accuracy = self.local_model.validate()
                resp['valid_loss'] = valid_loss
                resp['valid_accuracy'] = valid_accuracy

            self.sio.emit('client_update', resp)

        def on_stop_and_eval(*args):
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.sio.emit('client_eval', resp)

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)

        # TODO: later: simulate datagen for long-running train-serve service
        # i.e. the local dataset can increase while training

        # self.lock = threading.Lock()
        # def simulate_data_gen(self):
        #     num_items = random.randint(10, FederatedClient.MAX_DATASET_SIZE_KEPT * 2)
        #     for _ in range(num_items):
        #         with self.lock:
        #             # (X, Y)
        #             self.collected_data_train += [self.datasource.sample_single_non_iid()]
        #             # throw away older data if size > MAX_DATASET_SIZE_KEPT
        #             self.collected_data_train = self.collected_data_train[-FederatedClient.MAX_DATASET_SIZE_KEPT:]
        #             print(self.collected_data_train[-1][1])
        #         self.intermittently_sleep(p=.2, low=1, high=3)

        # threading.Thread(target=simulate_data_gen, args=(self,)).start()

    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))


# possible: use a low-latency pubsub system for gradient update, and do "gossip"
# e.g. Google cloud pubsub, Amazon SNS
# https://developers.google.com/nearby/connections/overview
# https://pypi.python.org/pypi/pyp2p

# class PeerToPeerClient(FederatedClient):
#     def __init__(self):
#         super(PushBasedClient, self).__init__()    

def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)


def log_trns(df, col):
    return df[col].apply(np.log1p)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


if __name__ == "__main__":

    os.chdir('..')
    print(os.getcwd())
    #FederatedClient(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', help='ip address')
    args = parser.parse_args()
    ipAddress = args.ip
    #print('Client #', args.num)
    FederatedClient(1)


