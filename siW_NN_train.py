import sys
import os
import numpy as np
import h5py
import scipy.io
import tensorflow as tf
from keras import ops
import keras
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
import keras_tuner as kt

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import math

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, average_precision_score
import sys
sys.path.append("/hpcnfs/data/GN2/gmandana/take2project/clip_2_electric_boogaloo/analysis/NN/Neural_Network_DNA_Demo")
sys.path.append("/hpcnfs/data/GN2/gmandana/bin/DeepRiPe_edited/Scripts/")
from helper import IOHelper, SequenceHelper

import random
rs = 1234
random.seed(rs)

def create_model1(num_task = 1,input_len_l = 500):
	K.clear_session()
	left_dim=4
	right_dim=4
	num_units=50
	input_l=input_len_l

	nb_f_l=[90,100]
	f_len_l=[7,7]
	p_len_l=[4,10]
	s_l=[2,5]
	nb_f_r=[90,100]
	f_len_r=[7,7]
	p_len_r=[10,10]
	s_r=[5,5]

	left_input = Input(shape=(input_l,left_dim),name="left_input")

	left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
	left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
	left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

	conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(left_drop1)
	merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
	merged_drop = Dropout(0.25)(merged_pool)
	merged_flat = Flatten()(merged_drop)

	hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=left_input, outputs=output)
	print(model.summary())
	return model

def model_builder(hp):

	num_task=1
	left_dim=4
	right_dim=4
	num_units=50
	input_l=500

	nb_f_l = hp.Int('filter_num', min_value=50, max_value=200, step=10)
	#nb_f_l=[90,100]
	f_len_l=hp.Int('filter_num', min_value=4, max_value=10, step=2)
	#f_len_l=[7,7]
	p_len_l=hp.Int('filter_num', min_value=4, max_value=10, step=2)
	#p_len_l=[4,10]
	s_l=hp.Int('filter_num', min_value=4, max_value=10, step=2)
	#s_l=[2,5]
	# Tune the learning rate for the optimizer
	# Choose an optimal value from 0.01, 0.001, or 0.0001

	##### architecture

	left_input = Input(shape=(input_l,left_dim),name="left_input")

	left_conv1 = Conv1D(filters=nb_f_l,kernel_size=f_len_l, padding='valid',activation="relu",name="left_conv1")(left_input)
	left_pool1 = MaxPooling1D(pool_size=p_len_l, strides=s_l,name="left_pool1")(left_conv1)
	left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

	#conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(left_drop1)
	#merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
	conv_merged = Conv1D(filters=nb_f_l,kernel_size= f_len_l, padding='valid',activation="relu",name="conv_merged")(left_drop1)
	merged_pool = MaxPooling1D(pool_size=p_len_l, strides=s_l)(conv_merged)
	merged_drop = Dropout(0.25)(merged_pool)
	merged_flat = Flatten()(merged_drop)

	hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=left_input, outputs=output)
	print(model.summary())

	model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])

	return model



def create_class_weight(labels_dict,total,mu=0.15):
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

@keras.saving.register_keras_serializable(package="my_package", name="myprecision")
def myprecision(y_true, y_pred):
	true_positives = tf.math.reduce_sum(tf.math.round(tf.keras.ops.clip(y_true * y_pred, 0, 1)))
	predicted_positives = tf.math.reduce_sum(tf.math.round(tf.keras.ops.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
	return precision

@keras.saving.register_keras_serializable(package="my_package", name="myrecall")
def myrecall(y_true, y_pred):
	true_positives = tf.math.reduce_sum(tf.math.round(tf.keras.ops.clip(y_true * y_pred, 0, 1)))
	possible_positives = tf.math.reduce_sum(tf.math.round(tf.keras.ops.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
	return recall

ntask = 1

### read in the sequence data
model_path = "/hpcnfs/data/GN2/gmandana/bin/restr_NN/"
fg_fasta_data = IOHelper.get_fastas_from_file("/hpcnfs/data/GN2/gmandana/bin/4.1.0/home/ieo5559/unifiedCLIP/NN_files/siW/fg.fa", uppercase=True)
bg_fasta_data = IOHelper.get_fastas_from_file("/hpcnfs/data/GN2/gmandana/bin/4.1.0/home/ieo5559/unifiedCLIP/NN_files/siW/bg.fa", uppercase=True)

sequence_length = len(fg_fasta_data.sequence[0])

fg_seq_matrix = SequenceHelper.do_one_hot_encoding(fg_fasta_data.sequence, sequence_length, SequenceHelper.parse_alpha_to_seq)
bg_seq_matrix = SequenceHelper.do_one_hot_encoding(bg_fasta_data.sequence, sequence_length, SequenceHelper.parse_alpha_to_seq)

seq_matrix = np.concatenate((fg_seq_matrix ,bg_seq_matrix))

### generate labels
Y = np.hstack([np.ones((np.shape(fg_seq_matrix)[0], )), np.zeros((np.shape(bg_seq_matrix)[0], ))])

X_train, X_test, y_train , y_test = train_test_split(seq_matrix, Y, test_size = 0.3, random_state = rs)

###### tuning
unique, counts = np.unique(y_train, return_counts=True)
labels_dict = dict(zip(unique, counts))
class_weight=create_class_weight(labels_dict, np.shape(y_train)[0], mu=0.5)

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='/hpcnfs/data/GN2/gmandana/bin/restr_NN/')

earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[earlystopper], class_weight = class_weight)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

########## build model with optimal hyperparams

model = tuner.hypermodel.build(best_hps)

checkpointer = ModelCheckpoint(filepath = os.path.join(model_path, "siW_NN_v2.keras"), verbose=1, save_best_only=True)

####### train

history = model.fit(X_train, y_train, 
    epochs=20, 
    batch_size=20,
    validation_data=(X_test,y_test), 
    class_weight=class_weight, verbose=2, 
    callbacks=[checkpointer,earlystopper])

model.save(os.path.join(model_path, "siW_NN_v2" + ".keras"))





