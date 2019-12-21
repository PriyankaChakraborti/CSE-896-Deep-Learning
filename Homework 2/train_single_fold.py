
# Import required modules
import numpy as np
import os
import tensorflow as tf
from util import one_hot_encode, split_data, plot_learning_curve
from model import *
from sklearn.metrics import classification_report
import pickle

##################### Chosen parameters #########################

# Choose model to use
#model_name = 'conv_model_1' # Has 4 convolutional layers
model_name = 'conv_model_2' # Has 8 convolutional layers
#model_name = 'autoencoder' # Run auto-encoder with imagenet dataset
#model_name = 'transfer_model' # Run auto-encoder, either pretraining if using imagenet or trained architecture for CIFAR

learning_rate = 0.0001 # Learning rate for Adam Optimizer
beta_1 = 0.99 # 1st momentum for Adam Optimizer (default is 0.99)
beta_2 = 0.999 # 2nd momentum for Adam Optimizer (default is 0.999)
epochs = 50
batch_size=50
kernel_size = 3

reg_scale = 0.001 # L2 regularization (Used 0.001 for initial tests)
use_early_stopping = False # Use early stopping or not
patience_no = 10 # Number of epochs to wait for improvement if early stopping turned on

# Select proportion of data into training set (1-train_test_prop) goes into test
train_test_prop = 0.8
optimization_perc = 0.9 # Percent of training data to go into optimization (rest goes into validation set)
##################################################################

# Names of classes
class_labels = list(range(100))
class_labels = [str(name) for name in class_labels]

# Run if you are loading CIFAR dataset and not using auto-encoder
if not (model_name == 'autoencoder'):

    # Load the CIFAR-100 dataset
    labels = np.load('/work/cse479/shared/homework/02/cifar_labels.npy')
    data = np.load('/work/cse479/shared/homework/02/cifar_images.npy')
    
    save_directory = './homework2_sessions'
    
    # Randomize order
    np.random.seed(42) # Seeded so we always get same splits
    idx = np.random.permutation(data.shape[0])
    data,labels = data[idx], labels[idx]
    # Reshape the data
    data = np.reshape(data, [-1, 32, 32, 3])
    
    # Save before one-hot-encoding
    raw_labels = [int(label) for label in labels]

    # One hot encode the labels
    labels = one_hot_encode(labels)

    # Split data into training and testing sets
    train_data, train_labels, test_data, test_labels = split_data(train_test_prop, data, labels)
    
    # Grab corresponding raw_labels for train and test
    _,train_labels_raw,_,raw_labels=split_data(train_test_prop,data,raw_labels)

    # Calculate the number of training and testing samples
    train_num_examples, test_num_examples = train_data.shape[0], test_data.shape[0]

# Load imagenet data for use with Autoencoder
elif model_name == 'autoencoder':
    
    save_directory = './homework2_sessions_autoencoder'
    
    # Load image data
    data = np.load('/work/cse479/shared/homework/02/imagenet_images.npy')
    print(data.shape)
    
    # Randomize order
    np.random.seed(42) # Seeded so we always get same splits
    idx = np.random.permutation(data.shape[0])
    data = data[idx]
    
    # Note for Imagenet train_data includes all data
    train_num_examples = data.shape[0]
    train_data = np.reshape(data, [-1, 32, 32, 3])
    print(train_data.shape)
    

def main_fun(batch_size, epochs, kernel_size, use_early_stopping, patience_no):
    ################# Run for CIFAR dataset ############################
    if not (model_name == 'autoencoder'):
        
        # Initialize empty lists
        avg_acc_val_list = []
        avg_ce_val_list = []
        avg_acc_train_list = []
        avg_ce_train_list = []
        best_epoch_acc_val = []
        best_epoch_ce_val = []
        
        # Print details of all chosen hyperparameters
        print('Model: {}'.format(model_name))
        print('Batch Size: ' + str(batch_size))
        print('Epochs: {}'.format(epochs))
        
        train_set,train_label,val_set,val_label=split_data(optimization_perc,train_data,train_labels)

        # Grab number of instances in training and validation sets
        num_train_data=train_set.shape[0]
        num_val_data=val_set.shape[0]
        
        print('We have %i training samples and have set aside %i for assessing early stopping' %(num_train_data,num_val_data))
        print('We have %i test samples' %test_num_examples)
        
        # We must reset graph for each new fold as weights must be distinct
        tf.compat.v1.reset_default_graph()
        with tf.Session() as session:

            # Only create input_tensor for convolutional network as it already exists within autoencoder graph
            if not (model_name == 'transfer_model'):
                input_tensor = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
            # Create output tensor regardless
            output_tensor = tf.compat.v1.placeholder(tf.float32, [None, 100], name='label_placeholder')
            
            if model_name == 'conv_model_1':
                confusion_matrix_op, cross_entropy, train_op, global_step_tensor, saver, accuracy, output = conv_model_1(input_tensor, output_tensor, kernel_size, reg_scale, learning_rate, beta_1, beta_2)
            elif model_name == 'conv_model_2':
                confusion_matrix_op, cross_entropy, train_op, global_step_tensor, saver, accuracy, output = conv_model_2(input_tensor, output_tensor, kernel_size, reg_scale, learning_rate, beta_1, beta_2)
            # Load modified autoencoder graph 
            elif model_name == 'transfer_model':
                confusion_matrix_op, cross_entropy, train_op, global_step_tensor, saver, accuracy, input_tensor, output = transfer_model(output_tensor, learning_rate, beta_1, beta_2, reg_scale, session)

            # Initialize all variables
            session.run(tf.compat.v1.global_variables_initializer())

            # Initialize empty lists
            test_loss=[]
            acc_overall=[]
            val_ce_overall = []

            # Initialize variables
            count=0
            best_validation_ce=float('inf')
            best_epoch=0

            # Go through desired number of training epochs
            for epoch in range(epochs):

                # Print the current epoch
                print('Epoch: {}'.format(str(epoch)))

                # Initialize empty lists for storing data for this current epoch
                loss_list_train = []
                conf_matrix_list_train = []
                accuracy_list_train = []
                loss_list_val = []
                conf_matrix_list_val = []
                accuracy_list_val = []

                #### Go through each minibatch for the training data ####
                tot_batches = num_train_data // batch_size
                for i in range(num_train_data // batch_size):

                    # Grab corresponding proportion of training data
                    batch_xs = train_set[i * batch_size:(i + 1) * batch_size, :]
                    batch_ys = train_label[i * batch_size:(i+1) * batch_size, :]

                    # Run this data through chosen model
                    _, loss_train, conf_matrix_train, accuracy_train = session.run([train_op, cross_entropy, confusion_matrix_op, accuracy], {input_tensor: batch_xs, output_tensor: batch_ys})

                    # Append cross entropy loss and confusion matrix predictions to lists
                    loss_list_train.append(loss_train)
                    conf_matrix_list_train.append(conf_matrix_train)
                    accuracy_list_train.append(accuracy_train)

                # Calculate average loss, regardless of which dataset or model was used
                avg_ce_train = sum(loss_list_train) / len(loss_list_train)
                # Calculate average accuracy
                avg_accuracy_train = sum(accuracy_list_train) / len(accuracy_list_train)

                # Append to lists for showing changes over folds
                avg_acc_train_list.append(avg_accuracy_train)
                avg_ce_train_list.append(avg_ce_train)

                #### Go through each minibatch for the validation data ####
                tot_batches = num_val_data // batch_size
                for i in range(num_val_data // batch_size):

                    # Grab corresponding proportion of training data
                    batch_xs = val_set[i * batch_size:(i + 1) * batch_size, :]
                    batch_ys = val_label[i * batch_size:(i+1) * batch_size, :]
                    # Run . this data through chosen model
                    loss_val, conf_matrix_val, accuracy_val = session.run([cross_entropy, confusion_matrix_op, accuracy], {input_tensor: batch_xs, output_tensor: batch_ys})

                    # Append cross entropy loss and confusion matrix predictions to lists
                    loss_list_val.append(loss_val)
                    conf_matrix_list_val.append(conf_matrix_val)
                    accuracy_list_val.append(accuracy_val)

                # Calculate average loss, regardless of which dataset or model was used
                avg_ce_val = sum(loss_list_val) / len(loss_list_val)
                # Calculate average accuracy
                avg_accuracy_val = sum(accuracy_list_val) / len(accuracy_list_val)
                
                print('Current validation accuracy: %f' %avg_accuracy_val)

                # Append to lists for showing changes over folds
                avg_acc_val_list.append(avg_accuracy_val)
                avg_ce_val_list.append(avg_ce_val)

                ##### Evaluate the Early Stopping Condition before continuing to next epoch #####
                if use_early_stopping: # This block is only run if we have chosen to implement early stopping
                    # If our new validation cross-entropy is smaller than previous run this
                    if(best_validation_ce > avg_ce_val):
                        # Save this epoch number
                        best_epoch=epoch
                        # Reset the count to 0, if it reaches set number we exit
                        count=0
                        # Redefine the best validation ce as the new one
                        best_validation_ce=avg_ce_val
                        # Save the session so we can reload it later
                        path_prefix=saver.save(session,os.path.join(save_directory,"homework_2"))

                    # This block is run if this epoch does not have a smaller validation ce than we have seen
                    else:
                        # Increment the count by 1, if it reaches set number we exit
                        count+=1

                    # This block is run if we have not seen an improvement in the set number of epochs
                    if count>= patience_no:
                        print("No improvement found during last iterations, stopping optimization.")
                        # Reload the session which was shown to be optimal
                        # loading the meta graph re-creates the graph structure in the current session
                        saver = tf.train.import_meta_graph(path_prefix + '.meta')
                        # This restores and initializes the saved variables
                        saver.restore(session, path_prefix) # restore session with the best ce

                        # Break out and do not continue investigating any additional epochs
                        break

            
            if use_early_stopping==True:
                # Before going to next fold append overall accuracy for the best epoch
                best_epoch_acc_val.append(avg_acc_val_list[best_epoch])
                # Before going to next fold append overall cross-entropy for the best epoch
                best_epoch_ce_val.append(avg_ce_val_list[best_epoch])

            # This is run if we have chosen not to implement Early Stopping
            else:
                # Save the session
                path_prefix=saver.save(session,os.path.join(save_directory,"homework_2"))

                # Append accuracy and ce for this fold to list
                best_epoch_acc_val.append(avg_acc_val_list[-1])
                best_epoch_ce_val.append(avg_ce_val_list[-1])
                
            ############## Evaluate On Test Set #################
            # Initialize empty lists
            ce_test=[]
            conf_test=[]
            acc_test=[]
            pred_labels_list = []
            conf_matrix_test = []
            classification_report_test = []
            
            # Go through each Mini-Batch
            for i in range(test_num_examples//batch_size):
                        
                # Grab subset of data and labels
                batch_xs= test_data[i*batch_size:(i+1)*batch_size,:]
                batch_ys=test_labels[i*batch_size:(i+1)*batch_size,:]
                
                # Run data through model
                test_ce, conf_matrix_test, accuracy_test,pred_labels = session.run([cross_entropy, confusion_matrix_op, accuracy, tf.argmax(output,1)], {input_tensor: batch_xs, output_tensor: batch_ys})
                
                # Append results to lists
                ce_test.append(test_ce)
                conf_test.append(conf_matrix_test)
                acc_test.append(accuracy_test)
                pred_labels_list.extend(pred_labels)
            
            # After all Mini-Batches complete calculate average values
            acc_test = sum(acc_test) / len(acc_test)
            ce_test = sum(ce_test) / len(ce_test)
            
            # Output values for accuracy and cros-entropy to user
            print('TEST TOTAL ACCURACY: ' + str(acc_test))
            print('TEST TOTAL LOSS: ' + str(ce_test))
            print('')
            
            # Output total test confusion matrix to user
            conf_matrix_test = sum(conf_test)
            print('TEST CONFUSION MATRIX:')
            print(str(conf_matrix_test))
            print('')
            
            # Calculate the classification report (including Recall, Precision, and F1-Score)
            classification_report_test = (classification_report(raw_labels,pred_labels_list,target_names=class_labels))
            # Output classification report to user
            print('')
            print('TEST CLASSIFICATION REPORT:')
            print(str(classification_report_test))
            
            ##### Show overall average accuracy #####
            print('The best epoch found is epoch ' + str(best_epoch))
            print('')
                      
        return avg_acc_val_list, avg_ce_val_list, acc_test, ce_test, avg_acc_train_list, avg_ce_train_list, conf_matrix_test,classification_report_test,raw_labels,pred_labels_list,best_epoch
    #########################################################################


    ##################### Run for autoencoder ###########################
    elif model_name == 'autoencoder':
        with tf.Session() as session:
            
            # Define the autoencoder network
            input_tensor = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
            total_loss, train_op, global_step_tensor, saver = autoencoder(input_tensor, learning_rate, beta_1, beta_2, reg_scale)

            # Initialize all variables
            session.run(tf.global_variables_initializer())

            # Print details of all chosen hyperparameters
            print('Model: {}'.format(model_name))
            print('Batch Size: {}'.format(batch_size))

            # Initialize empty lists for storing data for this current epoch
            loss_list_train = []

            # Go through each minibatch for the training data
            tot_batches = train_num_examples // batch_size
            for i in range(train_num_examples // batch_size):
                
                # Show progress every 1000 mini-batches
                if i % 100 == 0:
                    print('Looking at mini-batch %i out of %i' %(i, tot_batches))
                
                # Grab corresponding proportion of training data
                batch_xs = train_data[i * batch_size:(i + 1) * batch_size, :]

                # Run this data through the chosen model
                _, train_loss = session.run([train_op, total_loss], {input_tensor: batch_xs})

                # Append loss to the list
                loss_list_train.append(train_loss)

            # Save the model After Completion
            path_prefix  =saver.save(session,os.path.join(save_directory,"homework_2"))

            return loss_list_train
    ###########################################################

if not (model_name == 'autoencoder'):
    acc_val_list,ce_val_list,acc_test,ce_test, acc_train_list, ce_train_list, conf_matrix_test, classification_report_test,actual_labels_test,pred_labels_test,best_epoch = main_fun(batch_size, epochs, kernel_size, use_early_stopping, patience_no)

    # Create and plot the learning curve using training and validation sets
    plot_learning_curve(acc_train_list,acc_val_list,ce_train_list,ce_val_list)

    # Save the correct test class labels as pickle file for later analysis
    pickle_out = open("actual_labels_test.pickle","wb")
    pickle.dump(actual_labels_test, pickle_out)
    pickle_out.close()

    # Save predicted test class labels as pickle file for later analysis
    pickle_out = open("pred_labels_test.pickle","wb")
    pickle.dump(pred_labels_test, pickle_out)
    pickle_out.close()

elif model_name=='autoencoder':
    loss_list_train = main_fun(batch_size, epochs, kernel_size, use_early_stopping, patience_no)
    # Calculate average loss
    avg_train_loss = sum(loss_list_train) / len(loss_list_train) 
    print(avg_train_loss)
