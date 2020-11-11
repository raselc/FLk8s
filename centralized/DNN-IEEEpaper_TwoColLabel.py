from builtins import print

import pandas as pd
import numpy as np
import tensorflow as tf
import time
import gc

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import f1_score


def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)


def log_trns(df, col):
    return df[col].apply(np.log1p)


def build_network_Bin():
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


def dnnBinaryClass(rows):
    with open('kddcup.names.txt', 'r') as infile:
        kdd_names = infile.readlines()

    # kdd_cols holding the 41 features in the data set
    kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]

    # The Train+/Test+ data sets include sample difficulty rating and the attack class
    kdd_cols += ['class']
    # print(kdd.head())

    kdd_temp = pd.read_csv('KDDTrainForDummy.csv', names=kdd_cols)
    kdd = pd.read_csv('KDDTrain--.txt', names=kdd_cols, nrows=rows)
    kdd_t = pd.read_csv('KDDTest--.txt', names=kdd_cols)
    print("\n \n \n \n",kdd.shape[0],"\n \n \n")

    # kdd_full_cols holding all the values for the categorical features, total size: (122 + class)
    kdd_full_cols = [kdd_temp.columns[0]] + sorted(list(set(kdd_temp.protocol_type.values))) + sorted(
        list(set(kdd_temp.service.values))) + sorted(list(set(kdd_temp.flag.values))) + kdd_temp.columns[4:].tolist()

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

    # *** One-hot Encoding ***
    # for the cat features (in train and test sets): remove its original column and generate its dummy values
    cat_lst = ['protocol_type', 'service', 'flag']
    for col in cat_lst:
        kdd = cat_encode(kdd, col)
        kdd_t = cat_encode(kdd_t, col)
    # print(kdd['duration'].describe())

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
    target = kdd.pop('class')
    y_test = kdd_t.pop('class')

    # Generate dummy values for the label columns in train/test sets
    target = pd.get_dummies(target)
    y_test = pd.get_dummies(y_test)

    # target = train_y; train = train_x; test = test_x; y_test = test_y
    target = target.values
    train = kdd.values
    test = kdd_t.values
    y_test = y_test.values

    # We rescale features to [0, 1]
    min_max_scaler = MinMaxScaler()
    train = min_max_scaler.fit_transform(train)
    test = min_max_scaler.transform(test)

    print("Nb of input: ", train.shape[1])

    train, x_val, target, y_val = train_test_split(train, target, test_size=1 / 10, random_state=1)

    # for idx, col in enumerate(list(kdd.columns)):
    #    print(idx, col)

    NN = build_network_Bin()

    # We use early stopping on a holdout validation set
    # More info: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-
    # at-the-right-time-using-early-stopping/
    # patience: means that after 3 epochs in a row in which the model doesn't improve, training will stop
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

    ''''# ** Define a checkpoint callback **
    checkpoint_name = 'Experiments/Centralized/BestModelForTwoCol/weights.best.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [early_stopping, checkpoint]'''

    callbacks_list = [early_stopping]

    # In last experiment:
    # NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128, callbacks=[early_stopping])
    # batch size was 128, before moving to FL, after fixing nb param, after first results displayed
    history = NN.fit(x=train, y=target, epochs=5, validation_data=(x_val, y_val), batch_size=10)

    print(history.history.keys())
    # "Loss"
    '''plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()'''

    ''''# Load weights file of the best model :
    weights_file = 'Experiments/Centralized/BestModelForTwoCol/weights.best.hdf5'  # choose the best checkpoint
    NN.load_weights(weights_file)  # load it
    NN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])'''

    preds = NN.predict(test)
    pred_lbls = np.argmax(preds, axis=1)
    true_lbls = np.argmax(y_test, axis=1)

    test_loss, test_accuracy = NN.evaluate(test, y_test)

    print('Test Loss: ', test_loss)
    print('Test Accuracy: ', test_accuracy)

    # With the confusion matrix, we can aggregate model predictions
    # This helps to understand the mistakes and refine the model

    print(confusion_matrix(true_lbls, pred_lbls), '\n')
    print('Detection Accuracy:')
    print(1 - zero_one_loss(true_lbls, pred_lbls))
    gc.collect()

    # print(f1_score(true_lbls, pred_lbls, average='weighted')).

    # PS: root_shell continuous feature to be modified


def main():
    datasize = 100
    time.sleep(5)
    while(1):
        start_time = time.time()
        dnnBinaryClass(datasize)
        execTime = time.time() - start_time
        f = open("stats.txt","a+")
        dt = str(datasize) +"\t"+str(time.time())+"\t"+str(execTime)+"\n"
        f.write(dt)
        f.close()
        print('\nExecution Time: ', time.time() - start_time, "seconds")
        datasize = datasize + 100
        if datasize > 12000:
            break
        gc.collect()
        time.sleep(5)


if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run(main)
    main()
