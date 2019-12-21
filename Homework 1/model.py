
# coding: utf-8

# Import modules
import tensorflow as tf
import numpy as np

##### This holds functions which are called from models to calculate necessary information ######
def opt_metrics(labels,predictions,learning_rate,beta_1,beta_2,optimizer):
    """
    Args:
        - labels: Actual labels for each example
        - predictions: models predicted output
        - learning rate: rate for the Adam optimizer
        - beta 1: first momentum for Adam
        - beta 2: second momentum for Adam
    """
    
    # Define the cross-entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels , logits=predictions))
    
    # Collect the regularization losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    REG_COEFF = 1 # Note we decreased regularization in graph, so do not need to decrease here
    # this value is what we'll pass to `minimize`
    xentropy_w_reg = cross_entropy + REG_COEFF * sum(regularization_losses)
    
    # Define the confusion matrix
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(labels, axis=1), tf.argmax(predictions, axis=1), num_classes=10)

    # Set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)

    # Define the training optimizer
    train_op = optimizer.minimize(xentropy_w_reg, global_step=global_step_tensor)

    # Correct predictions
    correct = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))

    # Calculate Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # Define saver
    saver = tf.train.Saver()

    # Define Tensorboard, although it may not be used
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cross entropy', xentropy_w_reg)
    tf.summary.histogram("cross_entropy_hist", xentropy_w_reg)
    tf.summary.histogram("accuracy_hist", accuracy)
    merge = tf.summary.merge_all()

    return confusion_matrix_op, xentropy_w_reg, train_op, global_step_tensor, saver, accuracy, merge


##### Define two layer neural network #####
def TwoLayerNet(x, y, lr, b_1, b_2, l_size_1, l_size_2, reg_scale,optimizer,dropout_perc):
    """
     Args:
        - x: models feature_matrix
        - y: Actual labels for each example
        - lr: rate for the Adam optimizer
        - b_1: first momentum for Adam
        - b_2: second momentum for Adam
        - l_size_1: num neurons  for hidden layer 1
        - l_size_2: num neurons for hidden layer 2
        - reg: scale 0.0 for no regularization
    """

    # Normalize grayscale data
    x = x / 255.0

    # Define Model
    with tf.name_scope('two_layer_net') as scope:
        
        # first hidden layer with L2 regularization
        # Used variance_scaling_initializer as we use RELU
        hidden_1 = tf.layers.dense(x, l_size_1, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_1')

        # dropout 1
        dropout_1 = tf.layers.dropout(hidden_1, rate=dropout_perc, name='dropout_1')

        # second hidden layer with L2
        hidden_2 = tf.layers.dense(dropout_1, l_size_1, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_2')

        # dropout 2
        dropout_2 = tf.layers.dropout(hidden_2, rate=dropout_perc, name='dropout_2')

        # output layer with L2
        output = tf.layers.dense(dropout_2, 10,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 name='output_layer')
    
    # Name graph
    tf.identity(output, name='output')

    # Get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, merge= opt_metrics(y, output, lr, b_1, b_2,optimizer)
    
    return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, output, merge


##### Define three layer network #####
def ThreeLayerNet(x, y, lr, b_1, b_2, l_size_1, l_size_2, l_size_3,reg_scale,optimizer,dropout_perc):
    """
     Args:
        - x: models predicted output
        - y: Actual labels for each example
        - lr: rate for the Adam optimizer
        - b_1: first momentum for Adam
        - b_2: second momentum for Adam
        - l_size_1: layer size for hidden layers
        - l_size_2: another layer size for hidden layers
        - reg: scale 0.0 for no regularization
    """

    # Normalize grayscale data
    x = x / 255.0

    # Create Model
    with tf.name_scope('four_layer_net') as scope:
        
        # first hidden layer with L2 regularization
        # Used variance_scaling_initializer as we use RELU
        hidden_1 = tf.layers.dense(x, l_size_1, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_1')

        # first dropout layer
        dropout_1 = tf.layers.dropout(hidden_1, rate=dropout_perc, name='dropout_1')

        # second hidden layer with L2
        hidden_2 = tf.layers.dense(hidden_1, l_size_2, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_2')
        # second dropout layer
        dropout_2 = tf.layers.dropout(hidden_2, rate=dropout_perc, name='dropout_2')

        # third hidden layer with L2
        hidden_3 = tf.layers.dense(hidden_2, l_size_3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_3')
        # third dropout layer
        dropout_3 = tf.layers.dropout(hidden_3, rate=dropout_perc, name='dropout_3')

        # output layer with L2
        output = tf.layers.dense(dropout_3, 10,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 name='output_layer')

    tf.identity(output, name='output')

    # Get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, merge = opt_metrics(y, output, lr, b_1, b_2, optimizer)

    return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, output, merge

##### Define Four Layer Network #####
def FourLayerNet(x, y, lr, b_1, b_2, l_size_1, l_size_2, l_size_3,l_size_4,reg_scale,optimizer,dropout_perc):
    """
     Args:
        - x: models predicted output
        - y: Actual labels for each example
        - lr: rate for the Adam optimizer
        - b_1: first momentum for Adam
        - b_2: second momentum for Adam
        - l_size_1: layer size for hidden layers
        - l_size_2: another layer size for hidden layers
        - reg: scale 0.0 for no regularization
    """

    # Normalize grayscale data
    x = x / 255.0

    # Create Model
    with tf.name_scope('four_layer_net') as scope:
        
        # first hidden layer with L2 regularization
        # Used variance_scaling_initializer as we use RELU
        hidden_1 = tf.layers.dense(x, l_size_1, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_1')

        # first dropout layer
        dropout_1 = tf.layers.dropout(hidden_1, rate=dropout_perc, name='dropout_1')

        # second hidden layer with L2
        hidden_2 = tf.layers.dense(hidden_1, l_size_2, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_2')
        # second dropout layer
        dropout_2 = tf.layers.dropout(hidden_2, rate=dropout_perc, name='dropout_2')

        # third hidden layer with L2
        hidden_3 = tf.layers.dense(hidden_2, l_size_3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_3')
        # third dropout layer
        dropout_3 = tf.layers.dropout(hidden_3, rate=dropout_perc, name='dropout_3')

        # fourth hidden layer with L2
        hidden_4 = tf.layers.dense(hidden_3, l_size_4, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_4')

        # fourth dropout layer
        dropout_4 = tf.layers.dropout(hidden_4, rate=dropout_perc, name='dropout_4')

        # output layer with L2
        output = tf.layers.dense(dropout_4, 10,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 name='output_layer')
    
    # Name graph
    tf.identity(output, name='output')

    # Get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, merge = opt_metrics(y, output, lr, b_1, b_2, optimizer)

    return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, output, merge
