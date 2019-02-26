import numpy as np
from deepdrug3d import DeepDrug3DBuilder
import os
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.models import Sequential
from train_generator import DataGenerator
from valid_generator import V_DataGenerator
from keras.models import load_model
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 20} )
sess = tf.Session(config=config) 
keras.backend.set_session(sess)



os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_list = []
valid_list = []
label_list = []

train_folder = './train_data/'
valid_folder = './valid_data/'
output = './save_model/'
valid_num = 10

labels = {}

for file in os.listdir(train_folder):
    base = file.split('.')
    base_lst = base[0].split('_')
    if base_lst[4] == 'd':
        continue
    train_list.append(file[0:-6])
    labels[file[0:-6]] = base_lst[4]

for file in os.listdir(valid_folder):
    base = file.split('.')
    base_lst = base[0].split('_')
    valid_list.append(file[0:-6])
    labels[file[0:-6]] = base_lst[5]
# Datasets
# for file in os.listdir(train_folder):
#     base = file.split('.')
#     base_lst = base[0].split('_')
#     if base_lst[4] == 'd':
#         valid_list.append(file[0:-6])
#     else:
#         train_list.append(file[0:-6])
#     labels[file[0:-6]] = base_lst[4]


partition = {"train": train_list, "validation": valid_list}

batch_size = 8
epoch = 100
# Parameters
params = {'dim': (32, 32, 32),
          'n_channels': 28,
          'batch_size': batch_size,
          'n_classes': 2,
          'shuffle': True}

lr = 0.01
# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
print "the training data is ready"
validation_generator = V_DataGenerator(partition['validation'], labels, **params)
print "the validating data is ready"
model = DeepDrug3DBuilder.build()
model = multi_gpu_model(model,gpus=2)
print model.summary()
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# We add metrics to get more results you want to see
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

#tfCallBack = callbacks.TensorBoard(log_dir='./graph',
#                                   histogram_freq=0,
      #                             batch_size=batch_size,
#                                   write_graph=True,
 #                                  write_grads=False,
  #                                 write_images=True,
   #                                embeddings_freq=0,
    #                               embeddings_layer_names=None,
     #                              embeddings_metadata=None)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')




print "ready to fit generator"
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epoch,
                    verbose=2,
                    use_multiprocessing=True,
       #             callbacks=[tfCallBack],
		    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
  #                  validation_split=0.25,
                    workers=8)
# model.fit_generator(generator=training_generator,
      #              validation_data=validation_generator,
     #               epochs=epoch,
    #                verbose=2,
   #                 use_multiprocessing=True,
  #     #             callbacks=[tfCallBack],
 #                   callbacks=[earlyStopping, mcp_save, reduce_lr_loss],

#                    workers=8)


if output == None:
    model.save('deepdrug3d.h5')
else:
    if not os.path.exists(output):
        os.mkdir(output)
    if os.path.exists('deepdrug3d.h5'):
        os.remove('deepdrug3d.h5')
    model.save(output + 'deepdrug3d.h5')
    mm = load_model(output + 'deepdrug3d.h5')
    print mm.summary()
