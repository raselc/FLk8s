import pickle
import keras
import uuid
import os
import pandas as pd
import csv
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, Embedding, Flatten
from operator import itemgetter

from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
import msgpack
import random
import codecs
import numpy as np
import json
import msgpack_numpy
# https://github.com/lebedov/msgpack-numpy

import sys
import time

from sklearn.preprocessing import MinMaxScaler
from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/

import socket

round_to_start = 0


class GlobalModel(object):
    """docstring for GlobalModel"""

    def __init__(self):
        #self.graph = tf.get_default_graph()
        self.graph = tf.compat.v1.get_default_graph()
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        # for convergence check
        self.prev_train_loss = None

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        self.training_start_time = int(round(time.time()))

    def build_model(self):
        raise NotImplementedError()

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)

        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        self.current_weights = new_weights

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                           for i in range(len(client_sizes)))
        aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                                 for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.valid_losses += [[cur_round, cur_time, aggr_loss]]
        self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies
        }


class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):

        if round_to_start != 0:
            model = load_model('model.h5')
        else:
            # ~5MB worth of parameters
            model = Sequential()
            model.add(Dense(288, input_dim=121))
            model.add(Activation('tanh'))
            model.add(Dropout(.01))
            model.add(Dense(120))
            model.add(Activation('relu'))
            model.add(Dropout(.01))

            model.add(Dense(2))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model


# *** Flask server with Socket IO ***

# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):
    # TODO: change the settings
    MIN_NUM_WORKERS = 1
    MAX_NUM_ROUNDS = 5
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1
    ROUNDS_BETWEEN_VALIDATIONS = 2
    start_time = 0
    x_test = []
    y_test = []
    max_accuracy = 0

    def __init__(self, global_model, host, port):
        self.global_model = global_model()stats.txt

        # self.ready_client_sids = set()
        self.ready_client = pd.DataFrame(columns=('sID', 'TimeZone', 'Client_ID'))
        self.client_sids_selected_resources = list()
        self.client_sids_randomly_selected = list()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        self.model_id = str(uuid.uuid4())

        test_file_name = 'print.txt'
        self.write_all = open(test_file_name, 'w')

        #####
        # training states
        self.current_round = round_to_start  # 0 for not yet started
        self.completed_rounds = 0
        self.current_round_client_resources = []
        self.current_round_client_updates = []
        self.current_round_all_client_updates = []
        self.all_clients_resources = []
        self.eval_client_updates = []
        #####

        # socket io messages
        self.register_handles()

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, "disconnected")
            # if request.sid in self.ready_client_sids:
            #     self.ready_client_sids.remove(request.sid)
            if not self.ready_client[self.ready_client['sID'] == request.sid].empty:
                index_id = self.ready_client[self.ready_client['sID'] == request.sid].index
                self.ready_client.drop(index_id, inplace=True)
                self.ready_client.reset_index(drop=True, inplace=True)

        # TODO:
        @self.socketio.on('client_wake_up')
        def handle_wake_up(data):
            #TODO: resource prediction,
            data['client_time_zone'] = 'N'
            new_client = [request.sid, data['client_time_zone'], int(data['client_ID'])]
            self.ready_client.loc[len(self.ready_client)] = new_client

            if self.ready_client.__len__() >= FLServer.MIN_NUM_WORKERS and self.current_round == round_to_start:
                # add the rest of the clients (=90) to the list
                # TODO: change the range
                for i in range(2, 101):
                    data = dict()
                    data['client_time_zone'] = 'N'
                    data['client_ID'] = i

                    new_client = [0000, data['client_time_zone'], data['client_ID']]
                    self.ready_client.loc[len(self.ready_client)] = new_client

                self.write_all.write(str(self.ready_client))

                self.start_time = time.time()
                self.filter_client_stratified_for_comparison()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            # print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", data['client_ID'])

            '''for x in data:
                if x != 'weights':
                    print(x, data[x])'''

            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_all_client_updates += [data]

                if data['client_ID'] in self.client_sids_selected_resources:
                    self.current_round_client_updates += [data]
                    self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])

                if len(self.current_round_all_client_updates) == len(self.client_sids_randomly_selected):

                    result = sum(
                        item in self.client_sids_selected_resources for item in self.client_sids_randomly_selected)

                    # tolerate 10% unresponsive clients
                    if FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND - result > \
                            FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .3:
                        print('\n', (FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND - result), 'out of',
                              FLServer.MIN_NUM_WORKERS, 'Clients were selected, without being able to finish training')
                        if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                            self.discard_round_and_continue()
                            print('------------')
                            print(self.completed_rounds, 'out of', self.current_round, 'rounds successfully completed')
                            print('------------ \nDONE')
                            self.write_all.close()
                        else:
                            self.discard_round_and_continue()
                            self.train_next_round()
                    else:
                        self.completed_rounds += 1
                        print('\nStarting Aggregation for round', self.current_round,
                              'after receiving all needed updates..')
                        self.global_model.update_weights(
                            [x['weights'] for x in self.current_round_client_updates],
                            [x['train_size'] for x in self.current_round_client_updates],
                        )
                        aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                            [x['train_loss'] for x in self.current_round_client_updates],
                            [x['train_accuracy'] for x in self.current_round_client_updates],
                            [x['train_size'] for x in self.current_round_client_updates],
                            self.current_round
                        )

                        print("\naggr_train_loss", aggr_train_loss)
                        print("aggr_train_accuracy", aggr_train_accuracy, '\n')

                        if 'valid_loss' in self.current_round_client_updates[0]:
                            aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                                [x['valid_loss'] for x in self.current_round_client_updates],
                                [x['valid_accuracy'] for x in self.current_round_client_updates],
                                [x['valid_size'] for x in self.current_round_client_updates],
                                self.current_round
                            )
                            print("aggr_valid_loss", aggr_valid_loss)
                            print("aggr_valid_accuracy", aggr_valid_accuracy, '\n')

                        if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                            self.eval_on_server_and_continue()
                            print('------------')
                            print(self.completed_rounds, 'out of', self.current_round, 'rounds successfully completed')
                            print('------------ \nDONE')
                            self.write_all.close()

                        else:
                            self.eval_on_server_and_continue()
                            self.train_next_round()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            # tolerate 30% unresponsive clients
            # if len(self.eval_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
            # if len(self.eval_client_updates) == self.ready_client_sids.__len__():
            if len(self.eval_client_updates) == self.ready_client.__len__():
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                );
                print('Nb of clients contacted for testing: ', self.eval_client_updates.__len__())
                print("\naggr_test_loss", aggr_test_loss)
                print("aggr_test_accuracy", aggr_test_accuracy)
                print('\nExecution Time: ', time.clock() - self.start_time, "seconds")
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again

    def filter_client_stratified_for_comparison(self):

        # For server purpose: comparison to real devices behavior
        self.filter_client = self.ready_client.loc[self.ready_client.TimeZone == 'N']

        self.get_correct_clients_then_train()

    def get_correct_clients_then_train(self):

        #os.chdir("..")
        #os.chdir("..")
        #os.chdir("..")

        for index, row in self.filter_client.iterrows():
            client_train_file = 'Datasets/ClientsDS/Unbalanced/Train/Client_' + str(row['Client_ID']) + '.csv'
            client_res_file = 'Datasets/ClientsDS/Resources/Client_' + str(row['Client_ID']) + '.csv'

            train_df = pd.read_csv(client_train_file, header=None)
            res_df = pd.read_csv(client_res_file)

            data = dict()

            data['client_ID'] = row['Client_ID']
            data['train_and_valid_size'] = train_df.shape[0]
            data['data_size'] = list(res_df['DataSetSize'])
            data['cpu'] = list(res_df['CPU'])
            data['memory'] = list(res_df['RAM'])
            data['energy'] = list(res_df['Energy'])
            data['training_time'] = list(res_df['TrainingTime'])

            self.all_clients_resources += [data]

        # print(self.all_clients_resources)

        for x in self.all_clients_resources:

            res_df = pd.DataFrame(
                list(zip(x['data_size'], x['cpu'], x['memory'], x['energy'], x['training_time'])),
                columns=['DataSetSize', 'CPU', 'RAM', 'Energy', 'TrainingTime'])

            if self.sufficient_resources(x['client_ID'], res_df, x['train_and_valid_size']):
                print('Client', x['client_ID'], 'can participate')
                self.client_sids_selected_resources.append(x['client_ID'])
            else:
                print('Client', x['client_ID'], 'cannot participate')

        print('__________________________')
        print('# of clients able to participate in FL rounds: ',
              self.client_sids_selected_resources.__len__(), '\n')

        self.train_next_round()

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):

        self.current_round += 1

        # buffers all client updates
        self.current_round_client_updates = []
        self.current_round_all_client_updates = []

        print()
        print("### Round ", self.current_round, "###")
        self.client_sids_randomly_selected = random.sample(list(self.ready_client['Client_ID']),
                                                           FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        print("Requesting updates from: ", self.client_sids_randomly_selected.__len__(), '\n...\n..')

        to_print = "\n\n    Clients selected at round: " + str(self.current_round) + ":"
        self.write_all.write(to_print)
        self.write_all.write(str(self.client_sids_randomly_selected))

        '''print('\n........... Sleepinggg for 5 secs before starting Trainings .......\n')
        self.socketio.sleep(seconds=5)'''

        # by default each client cnn is in its own "room"
        temp_list = self.ready_client[self.ready_client['sID'] != 0]

        count = 0
        for rid in list(temp_list['sID']):
            if count < FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
                emit('init', {
                    'model_json': self.global_model.model.to_json(),
                    'model_id': self.model_id,
                    'epoch_per_round': 5,
                    'batch_size': 10,
                    'real_client_ID': self.client_sids_randomly_selected[count],
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=rid)

            count += 1

    def stop_and_eval(self):
        self.eval_client_updates = []
        for rid in list(self.ready_client['ID']):
            emit('stop_and_eval', {
                'model_id': self.model_id,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle'
            }, room=rid)

    def eval_on_server_and_continue(self):

        with self.global_model.graph.as_default():
            final_model = model_from_json(self.global_model.model.to_json())
            final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            final_model.set_weights(self.global_model.current_weights)

            if len(self.x_test) == 0:
                self.x_test, self.y_test = self.get_test_data()

            # score = self.global_model.model.evaluate(x_test, y_test, verbose=0)
            score = final_model.evaluate(self.x_test, self.y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            this_round_acc = score[1] * 100

            # One File for the 3 approaches
            file_name = 'Results_' + str(self.MIN_NUM_WORKERS) + 'workers.csv'
            acc_file_name = 'max_accuracy.txt'

            # Condition: Process was corrupted and start from where we stopped (i.e. round = 300)
            if round_to_start != 0 and self.current_round == round_to_start + 1:
                read_f = open(acc_file_name, 'r')
                self.max_accuracy = float(read_f.read())

            if this_round_acc < self.max_accuracy:
                this_round_acc = self.max_accuracy
            else:
                self.max_accuracy = this_round_acc
                final_model.save('model.h5')
                print("Saved model to disk")
                write_f = open(acc_file_name, 'w')
                write_f.write(str(self.max_accuracy))

            nb_clients = sum(item in self.client_sids_selected_resources for item in self.client_sids_randomly_selected)

            with open(file_name, 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)
                row = [self.current_round, lines[self.current_round][1], "%.2f" % this_round_acc,
                       lines[self.current_round][3], lines[self.current_round][4], nb_clients,
                       lines[self.current_round][6], lines[self.current_round][7]]
                lines[self.current_round] = row

            with open(file_name, 'w', newline='') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(lines)

            readFile.close()
            writeFile.close()

            return score[1]

    def discard_round_and_continue(self):

        acc_file_name = 'max_accuracy.txt'

        # Condition: Process was corrupted and start from where we stopped (i.e. round = 300)
        if round_to_start != 0 and self.current_round == round_to_start + 1:
            read_f = open(acc_file_name, 'r')
            self.max_accuracy = float(read_f.read())

        print('\nRound DISCARDED!')

        file_name = 'Results_' + str(self.MIN_NUM_WORKERS) + 'workers.csv'

        nb_clients = sum(item in self.client_sids_selected_resources for item in self.client_sids_randomly_selected)

        with open(file_name, 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)
            row = [self.current_round, lines[self.current_round][1], '-', lines[self.current_round][3],
                   lines[self.current_round][4], nb_clients, lines[self.current_round][6], lines[self.current_round][7]]
            lines[self.current_round] = row

        with open(file_name, 'w', newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)

        readFile.close()
        writeFile.close()

    def res_util_prediction(self, x_data_size, y_res_util, new_data_to_predict):
        x_mean = sum(x_data_size) / len(x_data_size)
        y_mean = sum(y_res_util) / len(y_res_util)

        alpha_num = 0
        alpha_denom = 0

        for i in range(len(x_data_size)):
            alpha_num += (x_data_size[i] - x_mean) * (y_res_util[i] - y_mean)
            alpha_denom += pow((x_data_size[i] - x_mean), 2)

        alpha = alpha_num / alpha_denom
        beta = y_mean - alpha * x_mean

        return alpha * new_data_to_predict + beta

    def get_test_data(self):

        with open('Datasets/kddcup.names.txt',
                  'r') as infile:
            kdd_names = infile.readlines()

        # kdd_cols holding the 41 features in the data set
        kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]

        # The Train+/Test+ data sets include sample difficulty rating and the attack class
        kdd_cols += ['class']

        kdd_temp = pd.read_csv('Datasets/KDDTrainForDummy.csv', names=kdd_cols)
        kdd_t = pd.read_csv('Datasets/KDDTest--.txt', names=kdd_cols)

        # kdd_full_cols holding all the values for the categorical features, total size: (122 + class)
        kdd_full_cols = [kdd_temp.columns[0]] + sorted(list(set(kdd_temp.protocol_type.values))) + sorted(
            list(set(kdd_temp.service.values))) + sorted(list(set(kdd_temp.flag.values))) + kdd_temp.columns[
                                                                                            4:].tolist()
        # for the cat features (in train and test sets): remove its original column and generate its dummy values
        cat_lst = ['protocol_type', 'service', 'flag']
        for col in cat_lst:
            kdd_t = cat_encode(kdd_t, col)

        # su_attempted column has a max value of 2.0, and is supposed to be binary feature.
        # Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0
        kdd_t['su_attempted'].replace(2, 0, inplace=True)

        # Next, we notice that the num_outbound_cmds column only takes on one value!
        # Now, that's not a very useful feature - let's drop it from the data set
        kdd_t.drop('num_outbound_cmds', axis=1, inplace=True)
        kdd_full_cols.remove('num_outbound_cmds')

        # np.log1p is used to correct the skew in the columns
        log_lst = ['duration', 'src_bytes', 'dst_bytes']
        for col in log_lst:
            kdd_t[col] = log_trns(kdd_t, col)
        # print(kdd['duration'].describe())

        for col in kdd_full_cols:
            if col not in kdd_t.columns:
                kdd_t[col] = 0

        kdd_t.columns = map(str.lower, kdd_t.columns)
        kdd_t = kdd_t.reindex(sorted(kdd_t.columns), axis=1)

        # Remove the 'class' column from train/test sets
        y_test = kdd_t.pop('class')

        # Generate dummy values for the label columns in train/test sets
        y_test = pd.get_dummies(y_test)

        test_x = kdd_t.values
        test_y = y_test.values

        # We rescale features to [0, 1]
        min_max_scaler = MinMaxScaler()
        test_x = min_max_scaler.fit_transform(test_x)

        return test_x, test_y

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port, debug=False)

    def sufficient_resources(self, client_ID, res_df, new_data):
        # Budget for data size > 2000
        budget_cpu = 100000  # 94.257
        # Budget for data size > 2100
        budget_memory = 100000  # 701.04
        # no budget
        budget_energy = 100000
        # Budget for data size > 1900
        time_threshold = 100000  # 231.66
        time_download = 0
        time_upload = 0

        predicted_cpu_util = self.res_util_prediction(res_df['DataSetSize'], res_df['CPU'], new_data)
        predicted_memory_util = self.res_util_prediction(res_df['DataSetSize'], res_df['RAM'], new_data)
        predicted_energy_util = self.res_util_prediction(res_df['DataSetSize'], res_df['Energy'], new_data)
        predicted_training_time_util = self.res_util_prediction(res_df['DataSetSize'], res_df['TrainingTime'], new_data)

        if predicted_cpu_util > budget_cpu or predicted_memory_util > budget_memory or \
                predicted_energy_util > budget_energy or (
                time_download + predicted_training_time_util + time_upload > time_threshold):
            about_resources = '** Client ' + str(client_ID) + ': '

            if predicted_cpu_util > budget_cpu:
                about_resources += ' X CPU '

            if predicted_memory_util > budget_memory:
                about_resources += ' X Memory '

            if predicted_energy_util > budget_energy:
                about_resources += ' X Energy '

            if time_download + predicted_training_time_util + time_upload > time_threshold:
                about_resources += ' X Time '

            about_resources += ' **'
            print(about_resources)

        return predicted_cpu_util < budget_cpu and predicted_memory_util < budget_memory and \
               predicted_energy_util < budget_energy and (
                       time_download + predicted_training_time_util + time_upload < time_threshold)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)


def log_trns(df, col):
    return df[col].apply(np.log1p)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.

    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)

    server = FLServer(GlobalModel_MNIST_CNN, "0.0.0.0", 5000)
    print("listening on, ", IPAddr);
    server.start()
