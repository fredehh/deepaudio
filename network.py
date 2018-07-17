import numpy as np
import datetime
import tensorflow as tf
import os
import sys
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from batchloader import batchloader

"""
General Parameters
"""

path = 'resources/data'  # Path to data
save_path = 'Models'  # Path where to save model
load_path = 'Models/nothing'  # Path to model
path_to_annotations = 'resources/'  # Path to annotations
path_to_specs = 'resources/data'  # Path to spectrograms
show_dimensions = True  # Print dimensions of data as passed through network (Sanity check)

load_model = False  # Load pretrained model
save_model = True  # Save model after training

name = '_stft'
regularization = True
reg_scale = 0.001  # Regularization and how much
reg_type = 'l2'  # Type of regularization 'l2' or 'l1'
dropout = False
keep_chance = 0.5  # Dropout and at what chance
batch_size = 32  # If tensorflow returns GPU memory related errors, reduce this
max_epochs = 50  # Amount of epochs to run
valid_every = 100  # How many batches we train before validation
seed = None  # Set seed for randomness to control output
GPU_FRAC = 0.85  # How much of GPU memory to allocate for training (Should be < 1)
optimizer_ = 'adam'  # adam or gradient (adam does not work with dropout)

tf.reset_default_graph()

height, width, nchannels = 221, 3007, 1  # Dimension of inputs
padding = 'same'  # Padding scheme

"""
Parameters for layers
"""

filters = [180]
kernel_sizes = [(221, 250)]
kernel_strides = [(221, 125)]
pool_sizes = [(1, 1)]
stride_pools = [(1, 1)]
units_fflayer = [256, 256]

"""
Checks for parameters
"""

same_length = False
reg_ok = False
opti_ok = False

if len(filters) == len(kernel_sizes) == len(kernel_strides) \
        == len(pool_sizes) == len(stride_pools):
    same_length = True

if not same_length:
    raise Exception("Length of CNN parameters do not match")

if reg_type == 'l1' or reg_type == 'l2':
    reg_ok = True

if not reg_ok:
    raise Exception("Unknown regularization type, must be 'l1' or 'l2'")

if optimizer_ == 'adam' or optimizer_ == 'gradient':
    opti_ok = True

if not opti_ok:
    raise Exception("Uknown optimizer, must be 'adam' or 'gradient'")

if save_model:
    save_dir = save_path + '/' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    os.makedirs(save_dir)
    file = open(save_dir + '/' + 'info.txt', 'w')

    # Output file
    file.write('Dropout: ' + str(dropout) + '\n')
    if dropout:
        file.write('Keep chance:../resources/audio_wavanced Machine Learning/data/ ' + str(keep_chance * 100) + '%\n')
    file.write('Regularization: ' + str(regularization) + '\n')
    if regularization:
        file.write('Reg scale: ' + str(reg_scale) + '\tReg type: ' + str(reg_type) + '\n')

left = tf.placeholder(tf.float32, [None, height, width, nchannels], name='Left')
right = tf.placeholder(tf.float32, [None, height, width, nchannels], name='Right')
label = tf.placeholder(tf.float32, [None, 2], name='Label')

if show_dimensions:
    print()
    print('input\t\t\t', left.get_shape())
    # Output file
    file.write('Input dimensions: ' + str(left.get_shape()) +
               '\nBatch Size: ' + str(batch_size) + '\n \n')
    file.write('Convlayer specs format:\n')
    file.write('Layername\tFilters\t\tKernel Size\tKernel Stride\t' +
               'Pool Size\tPool Stride\n')

"""
Create network
"""


def make_model(placeholder, filters, kernel_sizes, kernel_strides, pool_sizes,
               stride_pools, units_fflayer, show_dimensions, dropout,
               padding, first_model, reuse):
    for i in range(len(filters)):
        with tf.variable_scope('Convlayer' + str(i + 1), reuse=reuse):
            placeholder = conv2d(placeholder, filters[i], kernel_sizes[i],
                                 stride=kernel_strides[i], padding=padding, activation_fn=tf.nn.relu)

            # Output file
            if first_model:
                file.write('Convlayer' + str(i + 1) + '\t' + str(filters[i]) + '\t\t' + \
                           str(kernel_sizes[i]) + '\t\t' + str(kernel_strides[i]) + '\t\t' + str(pool_sizes[i]) + \
                           '\t\t' + str(stride_pools[i]) + '\n')

            if show_dimensions and first_model:
                print('output convlayer' + str(i + 1) + '\t', placeholder.get_shape())

    # Output file
    if first_model == True:
        file.write('\n')
        file.write('Denselayer hidden units:\n')
        file.write(str(units_fflayer) + '\n')

    return placeholder


file.write('\n')
file.write('Epochs completed \t Batches completed \t Training loss \t Training accuracy\n')

with tf.variable_scope("Siamese") as scope:
    model1 = make_model(left, filters, kernel_sizes, kernel_strides,
                        pool_sizes, stride_pools, units_fflayer,
                        show_dimensions, dropout, padding, True, False)
    scope.reuse_variables()
    model2 = make_model(right, filters, kernel_sizes, kernel_strides,
                        pool_sizes, stride_pools, units_fflayer,
                        show_dimensions, dropout, padding, False, True)

    placeholder = tf.concat([model1, model2], axis=1)

with tf.variable_scope("DenseLayer"):
    for i in range(len(units_fflayer)):
        with tf.variable_scope('Denselayer' + str(i + 1)):
            if i == 0:
                x = flatten(placeholder)
            x = fully_connected(x, units_fflayer[i], activation_fn=tf.nn.relu)

            if dropout:
                x = tf.layers.dropout(x, rate=keep_chance)

            if show_dimensions:
                print('output denselayer' + str(i + 1) + '\t', x.get_shape())

    with tf.variable_scope('Output_layer'):
        y = fully_connected(x, 2, activation_fn=tf.nn.softmax)
        if show_dimensions:
            print('final output\t\t', y.get_shape())
            print()
"""
Define loss, optimizer and accuracy
"""


def loss_func(y, label, regularization, reg_type):
    """Loss function for siamese network"""
    # loss = tf.reduce_sum(tf.divide(1,(1+tf.exp(-label*(y)))))
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=label))
    if regularization:
        if reg_type == 'l2':
            regularize = tf.contrib.layers.l2_regularizer(reg_scale)
        elif reg_type == 'l1':
            regularize = tf.contrib.layers.l1_regularizer(reg_scale)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = sum([regularize(param) for param in params])
        loss += reg_term
    # Return average loss
    return tf.reduce_mean(loss)


with tf.variable_scope('training'):
    # Define optimizer
    if optimizer_ == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    elif optimizer_ == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # Apply it
    with tf.name_scope("loss"):
        loss = loss_func(y, label, regularization, reg_type)
    train_op = optimizer.minimize(loss)

with tf.variable_scope('performance'):
    # Comparing results with labels
    y_acc = tf.one_hot(indices=tf.argmax(y, axis=1), depth=2)
    y_acc = tf.reshape(y_acc, [batch_size, 2])
    # Calculate accuracy
    accuracy = 1 - (tf.reduce_sum(tf.reduce_sum(tf.abs(y_acc - label), axis=1) / 2) / batch_size)

"""
Setup training parameters
"""

list_train = np.load(path_to_annotations + '/train.npy')
list_val = np.load(path_to_annotations + '/validation.npy')
list_test = np.load(path_to_annotations + '/test.npy')

# Force print statements to console
sys.stdout.flush()

valid_loss, valid_accuracy = [], []
train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []
saver = tf.train.Saver()
batches_completed = 0
epochs_completed = 0
epoch_complete = False
bc = 0  # Batch counter for indexing list
of = 0  # Offset for indexing list

# Temp batches to save executionbatch_size time
x_temp1 = np.zeros((batch_size, height, width, nchannels))
x_temp2 = np.zeros((batch_size, height, width, nchannels))

"""
Training loop
"""
# With GPU
# gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_FRAC)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
# Without GPU
with tf.Session() as sess:
    if load_model:
        try:
            saver.restore(sess, load_path)
            print("Model restored\n")
        except:
            print("No model found in:")
            print(load_path)
            print("Please put in correct load_path and try again\n")
            sys.exit()
    else:
        tf.initialize_all_variables().run()
    try:
        # progress = pb.ProgressBar(max_value=valid_every).start()
        print("\nEpochs completed \t Training loss \t Training acc \t Validation acc\n")
        sys.stdout.flush()
        while epochs_completed < max_epochs:
            _train_loss, _train_accuracy = [], []
            _valid_loss, _valid_acc = [], []
            # Run trainings
            temp_list = list_train[of + bc * batch_size:of + (bc + 1) * batch_size]
            if len(temp_list) == batch_size:
                bc += 1
            else:
                remain = len(list_train) % batch_size
                of = batch_size - remain
                temp_list = np.concatenate((list_train[-remain:], list_train[0:of]))
                bc = 0
                epochs_completed += 1
            song_left, song_right, y_batch = batchloader(name, temp_list,
                                                         path_to_specs, x_temp1, x_temp2)
            fetches_train = [train_op, loss, accuracy]
            feed_dict_train = {left: song_left, right: song_right,
                               label: y_batch}
            _, _loss, _acc = sess.run(fetches_train, feed_dict_train)
            _train_loss.append(_loss)
            _train_accuracy.append(_acc)
            batches_completed += 1
            # print("loss: %g" % (_loss))
            # progress.update(batches_completed % valid_every)

            # Compute validation loss and accuracy
            if batches_completed % valid_every == 0:
                dropout = False
                train_loss.append(np.mean(_train_loss))
                train_accuracy.append(np.mean(_train_accuracy))

                fetches_valid = [loss, accuracy]

                for i in range(len(list_val) // batch_size):
                    song_left, song_right, y_valid = \
                        batchloader(name, list_val[batch_size * i:batch_size * (i + 1)],
                                    path_to_specs, x_temp1, x_temp2)
                    feed_dict_valid = {left: song_left, right: \
                        song_right, label: y_valid}
                    _loss, _acc = sess.run(fetches_valid, feed_dict_valid)

                    _valid_loss.append(np.mean(_loss))
                    _valid_acc.append(np.mean(_acc))
                batch_left = len(list_val) % batch_size
                if batch_left != 0:
                    x_temp1 = np.zeros((batch_left, height, width, nchannels))
                    x_temp2 = np.zeros((batch_left, height, width, nchannels))
                    song_left, song_right, y_test = batchloader(name, list_val[-batch_left:],
                                                                path_to_specs, x_temp1, x_temp2)
                    feed_dict_valid = {left: song_left, right: song_right, label: y_test}
                    _loss, _acc = sess.run(fetches_valid, feed_dict_valid)
                    _valid_loss.append(np.mean(_loss))
                    _valid_acc.append(np.mean(_acc))

                valid_loss.append(np.mean(_valid_loss))
                valid_accuracy.append(np.mean(_valid_acc))
                dropout = True
                print("%d/%d:\t\t\t  %.3f\t\t  %.3f\t\t %.3f" \
                      % (epochs_completed + 1, max_epochs, train_loss[-1],
                         train_accuracy[-1], valid_accuracy[-1]))
                file.write(str(epochs_completed) + '\t\t' + str(batches_completed) + '\t\t' +
                           str(train_loss[-1]) + '\t\t' + str(train_accuracy[-1]) + '\n')
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass

    print('Training complete, testing accuracy')
    for i in range(len(list_test) // batch_size):
        dropout = False
        song_left, song_right, y_test = batchloader(name, list_test[batch_size * i:batch_size * (i + 1)],
                                                    path_to_specs, x_temp1, x_temp2)
        feed_dict_test = {left: song_left, right: song_right, label: y_test}
        _loss, _acc = sess.run(fetches_valid, feed_dict_test)
        test_loss.append(_loss)
        test_accuracy.append(_acc)
    left = len(list_test) % batch_size
    if left != 0:
        x_temp1 = np.zeros((left, height, width, nchannels))
        x_temp2 = np.zeros((left, height, width, nchannels))
        song_left, song_right, y_test = batchloader(name, list_test[-left:],
                                                    path_to_specs, x_temp1, x_temp2)
        feed_dict_test = {left: song_left, right: song_right, label: y_test}
        _loss, _acc = sess.run(fetches_valid, feed_dict_test)
        test_loss.append(_loss)
        test_accuracy.append(_acc)
    print('Test loss {:6.3f}, Test acc {:6.3f}'.format(
        np.mean(test_loss), np.mean(test_accuracy)))
    file.write('\n')
    file.write('Test loss \t Test accuracy')
    file.write(str(np.mean(test_loss)) + '\t\t' + str(np.mean(test_accuracy)))

    if save_model == True:
        save_path = save_dir + '/model/'
        saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

file.close()
