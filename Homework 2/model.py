import tensorflow as tf
import numpy as np

############### Define 1st Convolutional Architecture ###############
def conv_model_1(x, y, kernel_size, reg_scale, learning_rate, beta_1, beta_2):
    
    # Normalize the pixel values
    x = x / 255.0
    
    conv1_kernel = 5
    conv2_kernel = 5
    conv3_kernel = 3
    conv4_kernel = 3
    
    # Use for hyperparameter set 1
    conv1_filters = 32
    conv2_filters = 64
    conv3_filters = 128
    conv4_filters = 128
    dense1_neurons = 1024
    dense2_neurons = 512
    dense3_neurons = 256
    
    # Use for hyperparameter set 2
    #conv1_filters = 64
    #conv2_filters = 128
    #conv3_filters = 256
    #conv4_filters = 256
    #dense1_neurons = 2048
    #dense2_neurons = 1024
    #dense3_neurons = 512
    
    # Use for hyperparameter set 3
    #conv1_filters = 128
    #conv2_filters = 256
    #conv3_filters = 512
    #conv4_filters = 512
    #dense1_neurons = 4096
    #dense2_neurons = 2048
    #dense3_neurons = 512
    
    
    
    # Create Model
    with tf.name_scope('conv_model_1') as scope:
        ### Block 1 ###
        conv_1 = tf.compat.v1.layers.Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, padding='same', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale), name='conv_1')(x)
        leaky_relu_1 = tf.nn.leaky_relu(conv_1, name='leaky_relu_1')
        
        pool_1 = tf.compat.v1.layers.MaxPooling2D(2,2, padding='same')(leaky_relu_1)
        b_norm_1 = tf.compat.v1.layers.BatchNormalization(trainable=True)(pool_1)
        dropout_1 = tf.compat.v1.layers.Dropout(rate=0.8)(b_norm_1)
        ###############
        
        ### Block 2 ###
        conv_2 = tf.compat.v1.layers.Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_2')(dropout_1)
        leaky_relu_2 = tf.nn.leaky_relu(conv_2, name='leaky_relu_2')
        pool_2 = tf.compat.v1.layers.MaxPooling2D(2,2, padding='same')(leaky_relu_2)
        b_norm_2 = tf.compat.v1.layers.BatchNormalization(trainable=True)(pool_2)
        dropout_2 = tf.compat.v1.layers.Dropout(rate=0.8)(b_norm_2)
        ###############
        
        ### Block 3 ###
        conv_3 = tf.compat.v1.layers.Conv2D(filters=conv3_filters, kernel_size=conv3_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_3')(dropout_2)
        leaky_relu_3 = tf.nn.leaky_relu(conv_3, name='leaky_relu_3')
        dropout_3 = tf.compat.v1.layers.Dropout(rate=0.8)(leaky_relu_3)
        ###############
        
        ### Block 4 ###
        conv_4 = tf.compat.v1.layers.Conv2D(filters=conv4_filters, kernel_size=conv4_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_4')(dropout_3)
        leaky_relu_4 = tf.nn.leaky_relu(conv_4, name='leaky_relu_4')
        ###############

        # Flatten from 4D to 2D for dense layer
        flat = tf.reshape(leaky_relu_4, [-1, 8*8*conv4_filters])
        dense_1 = tf.compat.v1.layers.Dense(dense1_neurons, name='dense_1', activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(flat)
        dropout_4 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_1)
        dense_2 = tf.compat.v1.layers.Dense(dense2_neurons, name='dense_2', activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dropout_4)
        dropout_5 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_2)
        dense_3 = tf.compat.v1.layers.Dense(dense3_neurons, name='dense_3', activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dropout_5)
        dropout_6 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_3)

        #output
        output = tf.compat.v1.layers.Dense(100, name="output_layer", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dropout_6)

    tf.identity(output, name='output') 

    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy= opt_metrics(x, y, output, learning_rate, beta_1, beta_2)

    return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, output
################################################################################################

############### Define 2nd Convolutional Architecture ###############
def conv_model_2(x, y, kernel_size, reg_scale, learning_rate, beta_1, beta_2):
    
    # Normalize the pixel values
    x = x / 255.0
    
    conv1_kernel = 5
    conv2_kernel = 5
    conv3_kernel = 3
    conv4_kernel = 3
    
    # Use for hyperparameter set 1
    #conv1_filters = 32
    #conv2_filters = 64
    #conv3_filters = 128
    #conv4_filters = 128
    #dense1_neurons = 1024
    #dense2_neurons = 512
    #dense3_neurons = 256
    
    # Use for hyperparameter set 2
    #conv1_filters = 64
    #conv2_filters = 128
    #conv3_filters = 256
    #conv4_filters = 256
    #dense1_neurons = 2048
    #dense2_neurons = 1024
    #dense3_neurons = 512
    
    # Use for hyperparameter set 3
    conv1_filters = 128
    conv2_filters = 256
    conv3_filters = 512
    conv4_filters = 512
    dense1_neurons = 4096
    dense2_neurons = 2048
    dense3_neurons = 512
    
    # Create Model
    with tf.name_scope('conv_model_2') as scope:
        # block 1
        conv_1_1 = tf.compat.v1.layers.Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, padding='same', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale), name='conv_1_1')(x)
        conv_1_2 = tf.compat.v1.layers.Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, padding='same', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale), name='conv_1_2')(conv_1_1)
        leaky_relu_1 = tf.nn.leaky_relu(conv_1_2, name='leaky_relu_1')
        
        pool_1 = tf.compat.v1.layers.MaxPooling2D(2,2, padding='same')(leaky_relu_1)
        b_norm_1 = tf.compat.v1.layers.BatchNormalization(trainable=True)(pool_1)
        
        # block 2
        dropout_1 = tf.compat.v1.layers.Dropout(rate=0.8)(b_norm_1)
        conv_2_1 = tf.compat.v1.layers.Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_2_1')(dropout_1)
        conv_2_2 = tf.compat.v1.layers.Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_2_2')(conv_2_1)
        leaky_relu_2 = tf.nn.leaky_relu(conv_2_2, name='leaky_relu_2')
        
        pool_2 = tf.compat.v1.layers.MaxPooling2D(2,2, padding='same')(leaky_relu_2)
        b_norm_2 = tf.compat.v1.layers.BatchNormalization(trainable=True)(pool_2)

        # block 3
        dropout_2 = tf.compat.v1.layers.Dropout(rate=0.8)(b_norm_2)
        conv_3_1 = tf.compat.v1.layers.Conv2D(filters=conv3_filters, kernel_size=conv3_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_3_1')(dropout_2)
        conv_3_2 = tf.compat.v1.layers.Conv2D(filters=conv3_filters, kernel_size=conv3_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_3_2')(conv_3_1)
        leaky_relu_3 = tf.nn.leaky_relu(conv_3_2, name='leaky_relu_3')
        
        dropout_3 = tf.compat.v1.layers.Dropout(rate=0.8)(leaky_relu_3)
        conv_4_1 = tf.compat.v1.layers.Conv2D(filters=conv4_filters, kernel_size=conv4_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_4_1')(dropout_3)
        conv_4_2 = tf.compat.v1.layers.Conv2D(filters=conv4_filters, kernel_size=conv4_kernel, padding='same', 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),name='conv_4_2')(conv_4_1)
        leaky_relu_4 = tf.nn.leaky_relu(conv_4_2, name='leaky_relu_4')

        # flatten from 4D to 2D for dense layer
        flat = tf.reshape(leaky_relu_4, [-1, 8*8*conv4_filters])
        dense_1 = tf.compat.v1.layers.Dense(dense1_neurons, name='dense_1', activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(flat)
        dropout_4 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_1)
        dense_2 = tf.compat.v1.layers.Dense(dense2_neurons, name='dense_2', activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dropout_4)
        dropout_5 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_2)
        dense_3 = tf.compat.v1.layers.Dense(dense3_neurons, name='dense_3', activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dropout_5)
        dropout_6 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_3)

        #output
        output = tf.compat.v1.layers.Dense(100, name="output_layer", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dropout_6)

    tf.identity(output, name='output') 

    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy= opt_metrics(x, y, output, learning_rate, beta_1, beta_2)

    return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, output
################################################################################################

############### Define Autoencoder ###############
def autoencoder(x, learning_rate, beta_1, beta_2, reg_scale):
    """
        args: 
                - x: image batch of shape [?, 32, 32, 3]
                - learning_rate: learning rate
                - beta_1: momentum 1
                - beta_2: momentum 2

        returns: 

        """

    enc1_filt = 64
    enc2a_filt = 128
    enc2b_filt = 128
    enc3_filt = 256
    dec2_filt = 256
    dec3a_filt = 128
    dec3b_filt = 128
    dec6_filt = 64
    
    with tf.name_scope('autoencoder_scope') as scope:
    
        encoder_1 = tf.layers.Conv2D(enc1_filt, 3, 1, padding='same', name='encode_1', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(x)
        relu_1 = tf.nn.relu(encoder_1, name='relu_1')
        b_norm_1 = tf.layers.BatchNormalization(trainable=True)(relu_1)
        #output_shape=(16,16,64)

        encoder_2a = tf.layers.Conv2D(enc2a_filt, 3, 2, padding='same', name='encode_2a', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(b_norm_1)
        relu_2a = tf.nn.relu(encoder_2a, name='relu_2a')      
        b_norm_2a = tf.layers.BatchNormalization(trainable=True)(relu_2a)
        #output_shape(16,16,64)
        encoder_2b=tf.layers.Conv2D(enc2b_filt,5,1,padding='same',name='encode_2b',\
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(b_norm_1)
        pool_2b=tf.layers.MaxPooling2D(2, 2, padding='same')
        output_pool_2b=pool_2b(encoder_2b)
        relu_2b = tf.nn.relu(output_pool_2b, name='relu_2b')      
        b_norm_2b = tf.layers.BatchNormalization(trainable=True)(relu_2b)
        #now concatenate the two output_shape=(16,16,128)
        net = tf.concat(axis=3,values=[b_norm_2a,b_norm_2b], name='concat_encode')

        #output_shape:(8,8,128)
        encoder_3 = tf.layers.Conv2D(enc3_filt, 3, 2, padding='same', name='encode_3', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(net)
        relu_3 = tf.nn.relu(encoder_3, name='relu_3')
        b_norm_3 = tf.layers.BatchNormalization(trainable=True)(relu_3)
        flatten_dim = np.prod(b_norm_3.get_shape().as_list()[1:])       
        flat = tf.reshape(b_norm_3, [-1, flatten_dim], name='flat') 
        ##Code_Layer:output_shape=512,1
        code = tf.layers.Dense(512, activation=tf.nn.relu, name='code')(flat)
        decoder_1 = tf.reshape(code, [-1, 8, 8, 8], name='decode_1')

        #output_shape:(8,8,128)
        decoder_2 = tf.layers.Conv2D(dec2_filt, 3, 1, padding='same', name='decode_2')(decoder_1) 
        relu_4= tf.nn.relu(decoder_2, name='relu_4')
        b_norm_4 = tf.layers.BatchNormalization(trainable=True)(relu_4)
        #net layer:output_shape(16,16,128)
        #branch_1
        decoder_3a = tf.layers.Conv2DTranspose(dec3a_filt, 3,2, padding='same', name='decode_3a')(b_norm_4)   
        relu_5a = tf.nn.relu(decoder_3a, name='relu_5a')
        b_norm_5a = tf.layers.BatchNormalization(trainable=True)(relu_5a)
        #branch_2
        decoder_3b = tf.layers.Conv2DTranspose(dec3b_filt, 5,2, padding='same', name='decode_3b')(b_norm_4)   
        relu_5b = tf.nn.relu(decoder_3b, name='relu_5b')
        b_norm_5b = tf.layers.BatchNormalization(trainable=True)(relu_5b)
        net_dec=tf.concat(axis=3,values=[b_norm_5a,b_norm_5b], name='concat_decode')

        #output_shape(32,32,32)
        decoder_6 = tf.layers.Conv2DTranspose(dec6_filt, 3, 2, padding='same', name='decode_6', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(net_dec)
        relu_6 = tf.nn.relu(decoder_6, name='relu_6')
        b_norm_6 = tf.layers.BatchNormalization(trainable=True)(relu_6)
    
    #output_shape(32,32,1)
    output = tf.layers.Conv2DTranspose(3, 3, strides=(1,1), padding='same', name='output')(b_norm_6)
    
    return opt_metrics_autoencoder(x, code, output, learning_rate, beta_1, beta_2)
        
################################################################################################
        
############### Define Transfer Model built off Autoencoder Code layer ###############
def transfer_model(labels, learning_rate, beta_1, beta_2, reg_scale, session):
    """
        Args:
                - labels: Actual labels for each example        
                - lr: rate for the Adam optimizer
                - b_1: first momentum for Adam
                - b_2: second momentum for Adam
                - session: current tf session
        """
    # Note that the code is 4x4x8, so 128 neurons
    neuron_list = [1024, 512, 512, 256, 256]
    
    num_neurons_dense1 = neuron_list[0]
    num_neurons_dense2 = neuron_list[1]
    num_neurons_dense3 = neuron_list[2]
    num_neurons_dense4 = neuron_list[3]
    num_neurons_dense5 = neuron_list[4]

    # Import graph
    saver = tf.train.import_meta_graph('./homework2_sessions_autoencoder/homework_2.meta')
    # Restore model
    saver.restore(session, './homework2_sessions_autoencoder/homework_2')

    # Define the graph
    graph = session.graph
    
    #print(session.graph.get_operations())

    # Define input tensor based on graph
    x = graph.get_tensor_by_name('input_placeholder:0')

    # Grab the tensor just before the decoders
    code_layer = graph.get_tensor_by_name('code/Relu:0')

    # Now build the dense layers on top of this tensor
    with tf.name_scope('transfer_model') as scope:
        
        ##### USE BELOW IF STOPPING GRADIENT AFTER CODE LAYER #####
        # First stop any gradients from continuing backward, we don't train anything previously trained
        after_code_layer = tf.stop_gradient(code_layer)
        
        ##### USE BELOW IF RETRAINING ENTIRE NETWORK #####
        #after_code_layer = tf.identity(code_layer)
        
        # Dense Layer 1
        dense_1 = tf.compat.v1.layers.Dense(num_neurons_dense1, name="dense1", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(after_code_layer)
        dropout_1 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_1)

        # Dense Layer 2
        dense_2 = tf.compat.v1.layers.Dense(num_neurons_dense2, name="dense2", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dense_1)
        dropout_2 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_2)

        # Dense Layer 3
        dense_3 = tf.compat.v1.layers.Dense(num_neurons_dense3, name="dense3", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dense_2)
        dropout_3 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_3)
        
        # Dense Layer 4
        dense_4 = tf.compat.v1.layers.Dense(num_neurons_dense4, name="dense4", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dense_3)
        dropout_4 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_4)
        
        # Dense Layer 5
        dense_5 = tf.compat.v1.layers.Dense(num_neurons_dense5, name="dense5", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dense_4)
        dropout_5 = tf.compat.v1.layers.Dropout(rate=0.8)(dense_5)

        # Output Layer
        output = tf.compat.v1.layers.Dense(100, name="output_layer", activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))(dropout_5)

    ########################

    tf.identity(output, name="output")

    ###### This part of code defines optimizer for the added layers
    # Adam
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2,name='transf_adam')
    # training and saving functionality
    global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)

    # Cross entropy loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output))

    # Collect the regularization losses
    regularization_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES,'transfer_model')

    # Weight of the regularization for the final loss
    REG_COEFF = 0.1

    # value passes to minimize
    xentropy_w_reg = cross_entropy + REG_COEFF * sum(regularization_losses)
    # We get error with below line
    train_op = optimizer.minimize(xentropy_w_reg, global_step=global_step_tensor)

    ################


    # confusion matrix
    confusion_matrix_op  = tf.math.confusion_matrix(tf.argmax(labels, axis=1), tf.argmax(output, axis=1), num_classes=100)

    # optimizer
    transfer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'transfer_model')

    session.run(tf.variables_initializer(transfer_vars, name='init'))

    # correct predictions
    correct = tf.equal(tf.argmax(output, axis=1), tf.argmax(labels, axis=1))

    # accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # saver
    saver = tf.compat.v1.train.Saver()
    ####################

    return confusion_matrix_op, cross_entropy, train_op, global_step_tensor, saver, accuracy, x, output
        
################################################################################################

############### Definition Holding Metrics For the Convolutional Architectures ###############
def opt_metrics(x, labels, predictions, learning_rate, beta_1, beta_2):
    
    # cross entropy loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions))

    # collect the regularization losses
    regularization_losses= tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

    # weight of the regularization for the final loss
    REG_COEFF = 0.1

    # value passes to minimize
    xentropy_w_reg = cross_entropy + REG_COEFF * sum(regularization_losses)

    # confusion matrix
    confusion_matrix_op  = tf.math.confusion_matrix(tf.argmax(labels, axis=1), tf.argmax(predictions, axis=1), num_classes=100)

    # training and saving functionality
    global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)

    # Adam
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2)

    # optimizer
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(xentropy_w_reg, global_step=global_step_tensor)
    train_op = tf.group([train_op, update_ops])

    # correct predictions
    correct = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))

    # accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # saver
    saver = tf.compat.v1.train.Saver()

    return confusion_matrix_op, xentropy_w_reg, train_op, global_step_tensor, saver, accuracy
################################################################################################

############### Definition Holding Metrics For the Autoencoder Architecture ###############
def opt_metrics_autoencoder(x, code, output, learning_rate, beta_1, beta_2):
        
        # calculate loss
        sparsity_weight = 5e-3
        sparsity_loss = tf.norm(code, ord=1, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.square(output - x)) # Mean Square Error
        total_loss = reconstruction_loss + sparsity_weight * sparsity_loss
        # total_loss = reconstruction_loss

        global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        # Adam
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2)
        
        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
        train_op = tf.group([train_op, update_ops])

        saver = tf.train.Saver() 

        return total_loss, train_op, global_step_tensor, saver
################################################################################################
