import numpy as np
from deepdrug3d import DeepDrug3DBuilder
import os
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential
from train_generator import DataGenerator
from valid_generator import V_DataGenerator
from keras.models import load_model

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

batch_size = 2
epoch = 2
# Parameters
params = {'dim': (32, 32, 32),
          'n_channels': 28,
          'batch_size': batch_size,
          'n_classes': 2,
          'shuffle': True}

lr = 0.1
# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
print "the training data is ready"
validation_generator = V_DataGenerator(partition['validation'], labels, **params)
print "the validating data is ready"
model = DeepDrug3DBuilder.build()
print model.summary()
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# We add metrics to get more results you want to see
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

tfCallBack = callbacks.TensorBoard(log_dir='./graph',
                                   histogram_freq=0,
                                   batch_size=batch_size,
                                   write_graph=True,
                                   write_grads=False,
                                   write_images=True,
                                   embeddings_freq=0,
                                   embeddings_layer_names=None,
                                   embeddings_metadata=None)
print "ready to fit generator"
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epoch,
                    verbose=2,
                    use_multiprocessing=True,
                    callbacks=[tfCallBack],

                    workers=6)

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
