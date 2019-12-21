
# coding: utf-8

# This is used for iterating through a grid-search of architectures and hyperparameters

# Load modules
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import os
import argparse
import tensorflow as tf
from sklearn.model_selection import KFold
import itertools
import pandas as pd
import pickle
from model import TwoLayerNet,FourLayerNet
from util import split_data,one_hot_encode,generate_adam_param


#################################### Chosen Parameters For This Grid-Search #####################

# Initialize global variables
global epochs,patience_no,batch_size

# Below are the variables we set for this grid-search
num_folds = 5 # Choose 5-fold cross-validation
epochs = 50 # Go up only to 50 epochs
use_earlyStopping = False # For this grid-search we did not implement early stopping
patience_no=20 # Would have been used with early stopping if implemented
batch_size=50 # We used mini-batches of size 50
rate = 0.001 # Chosen as this is the default for Adam Optimizer and we want a fair comparison
prop=0.8 # Keep the 80% in train and 20% in test

# Below are the lists which we iterated over during the grid-search
neuron_list = list(range(100,400,100)) # We checked last hidden layer with 100, 200, and 300 neurons
dropout_list = [0,0.3] # Checked no dropout and 30% dropout
layers_list = [2,4] # We checked 2 and 4 layer networks
beta_1_list = [0.1,0.9] # For exponential decay rate for the first momentum estimate in Adam Optimizer
####################################################################################################

# get data and process it
labels = np.load('/work/cse496dl/shared/homework/01/fmnist_train_labels.npy')
data = np.load('/work/cse496dl/shared/homework/01/fmnist_train_data.npy')

# Randomize order
np.random.seed(42) # Seeded so we always get same splits
idx = np.random.permutation(data.shape[0])
data,labels = data[idx], labels[idx]

# This is where saved models are kept
save_directory = './homework1_sessions'

#one hot encode labels
labels=one_hot_encode(labels)

#split data into test and train
train_data,train_labels,test_data,test_labels=split_data(0.8,data,labels)

# Main definition for running the network
def main_FMNIST(num_layer,num_folds,epochs,num_neurons,optimizer,patience_number,batch_size,dropout_perc,use_earlyStopping=False):
    
    # Initialize K-Fold Cross validation
    kf=KFold(n_splits=num_folds,shuffle=False,random_state=None)
    # Get parameters for using the Adam Optimizer (here we use default)
    lr,b_1,b_2=generate_adam_param(default=True)
    # Initialize lists
    result_list = []
    result_ce_list = []
    
    # Go through each fold of the K-Fold Cross validation
    for train_idx,val_idx in kf.split(train_data,train_labels):
        
        # Select the indices of train and validation set for the folds
        train_set,val_set=train_data[train_idx],train_data[val_idx]
        train_label,val_label=train_labels[train_idx],train_labels[val_idx]
        
        # Grab number of instances in training and validation sets
        num_train_data=train_set.shape[0]
        num_val_data=val_set.shape[0]
        
        # We must reset graph for each new fold as weights must be distinct
        tf.reset_default_graph()
        with tf.Session() as session:
            
            # create tensorflow placeholders for data and labels
            input_tf = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
            output_tf = tf.placeholder(tf.float32, [None, 10], name='label_placeholder')
            
            # If we want the two-layer architecture run this block
            if num_layer==2:
                # We choose 3 times as many neurons in hidden layer 1 as hidden layer 2
                layer_size_1 = num_neurons*3
                layer_size_2 = num_neurons*1
                confusion_matrix_op_1, cross_entropy_1, train_op_1, global_step_tensor_1, saver_1,accuracy_1, output,merge_1 = TwoLayerNet(input_tf, output_tf, lr, b_1, b_2, layer_size_1, layer_size_2,0.0,optimizer,dropout_perc)
            # If we want the four-layer architecture run this block
            else:
                # We choose 3 times as many neurons in hidden layer 1 as hidden layer 4
                layer_size_1 = num_neurons*3
                layer_size_2 = int(num_neurons*2.5)
                layer_size_3 = int(num_neurons*1.5)
                layer_size_4 = num_neurons*1
                confusion_matrix_op_1, cross_entropy_1, train_op_1, global_step_tensor_1, saver_1, accuracy_1, output,merge_1 = FourLayerNet(input_tf, output_tf, lr, b_1, b_2, layer_size_1, layer_size_2,layer_size_3,layer_size_4,0.0,optimizer,dropout_perc)
            
            # Now initialize the variables in the tensorflow graph
            session.run(tf.global_variables_initializer())
            
            # Initialize empty lists
            test_loss=[]
            acc_overall=[]
            val_ce_overall = []
            
            # Initialize variables
            count=0
            best_validation_ce=float('inf')
            best_epoch=0
            
            # Go through each epoch of training
            for i in range(epochs):
                
                ##### Go through each minibatch for the optimization data #####
                
                # Initialize empty lists
                ce_train=[]
                conf_mxs=[]
                acc_train=[]
                for j in range(num_train_data//batch_size):
                    # Grab the data and labels for this minibatch
                    batch_xs=train_set[j*batch_size:(j+1)*batch_size,:]
                    batch_ys=train_label[j*batch_size:(j+1)*batch_size,:]
                    
                    # Run these through the graph
                    summary, _, train_ce, conf_matrix, accuracy = session.run([merge_1, train_op_1, cross_entropy_1, confusion_matrix_op_1, accuracy_1], {input_tf: batch_xs, output_tf: batch_ys})
                    
                    # Append the output to these lists
                    ce_train.append(train_ce)
                    conf_mxs.append(conf_matrix)
                    acc_train.append(accuracy)
                    
                # After we have run through all minibatches calculate the average training cross-entropy
                avg_train_ce = sum(ce_train) / len(ce_train)
                
                ##### Go through each minibatch for the optimization data #####
                
                # Initialize empty lists
                ce_val=[]
                conf_val=[]
                acc_val=[]
                
                for j in range(num_val_data//batch_size):
                    # Grab the data and labels for this minibatch
                    batch_xs=val_set[j*batch_size:(j+1)*batch_size,:]
                    batch_ys=val_label[j*batch_size:(j+1)*batch_size,:]
                    
                    # Run these through the graph
                    summary, val_ce, conf_matrix, accuracy = session.run([merge_1,cross_entropy_1, confusion_matrix_op_1, accuracy_1], {input_tf: batch_xs, output_tf: batch_ys})
                    
                    # Append the output to these lists
                    ce_val.append(val_ce)
                    acc_val.append(accuracy)
                
                # After we have run through all minibatches calculate the average values
                avg_validation_acc = sum(acc_val) / len(acc_val)
                avg_validation_ce = sum(ce_val) / len(ce_val)
                
                # Append these to the lists
                acc_overall.append(avg_validation_acc)
                val_ce_overall.append(avg_validation_ce)
                
                ##### Evaluate the Early Stopping Condition #####
                if use_earlyStopping: # This block is only run if we have chosen to implement early stopping
                    # If our new validation cross-entropy is smaller than previous run this
                    if(best_validation_ce > avg_validation_ce):
                        # Save this epoch number
                        best_epoch=i
                        # Reset the count to 0, if it reaches set number we exit
                        count=0
                        # Redefine the best validation ce as the new one
                        best_validation_ce=avg_validation_ce
                        # Save the session so we can reload it later
                        path_prefix=saver_1.save(session,os.path.join(save_directory,"homework_1"))
                        
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
                        saver_1.restore(session, path_prefix) # restore session with the best ce

                        # Break out from the loop
                        break
                        
            ##### This is done at end of each fold
            if use_earlyStopping==True:
                # Before going to next fold append overall accuracy for the best epoch
                result_list.append(acc_overall[best_epoch])
                # Before going to next fold append overall cross-entropy for the best epoch
                result_ce_list.append(val_ce_overall[best_epoch])
                
            # This is run if we have chosen not to implement Early Stopping
            else:
                # Save the session
                path_prefix=saver_1.save(session,os.path.join(save_directory,"homework_1"))
                
                # Append acceleration and ce for this fold to list
                result_list.append(acc_overall[-1])
                result_ce_list.append(val_ce_overall[-1])
                
                
    ##### This is run after all epochs have completed #####
    
    # Display the average accuracy over all folds
    print('The overall average accuracy over all folds is : %f' %np.mean(result_list))
    print('')
    return result_list,result_ce_list



############ This defines grid-search and runs all iterations through model ############

# Define pandas dataframem which will hold the results
df_grid_search = pd.DataFrame(columns=['Beta_1_Param','Num_Neurons','Dropout_Perc','Num_Layers','Acc_Fold_1','Acc_Fold_2','Acc_Fold_3','Acc_Fold_4','Acc_Fold_5','CE_Fold_1','CE_Fold_2','CE_Fold_3','CE_Fold_4','CE_Fold_5'])

# Create list of all combinations to check
grid_search_list = list(itertools.product(neuron_list,dropout_list,layers_list,beta_1_list))
# Calculate number of combinations
num_combinations = len(grid_search_list)

index=0
# Iterate through all combinations of these parameters
for num_neurons,dropout_rate,num_layers,beta_1_param in grid_search_list:
    
    # For user-friendliness print out progress
    print('')
    print('Checking combination ' + str(index+1) + ' out of ' + str(num_combinations))
    print('Using ' + str(num_layers) + ' layers, ' + str(num_neurons) + ' neurons per layer, momentum parameter ' + str(beta_1_param) + ', and dropout percentage ' + str(dropout_rate))
    print('')
    
    # Define Adam Optimizer
    optimizer = tf.train.AdamOptimizer(beta1=beta_1_param)
    
    # Run Parameters and collect the results
    result_list,ce_list=main_FMNIST(num_layers,num_folds,epochs,num_neurons,optimizer,patience_no,batch_size,
                                   dropout_rate,use_earlyStopping)
    
    # Append the results to the dataframe
    df_grid_search.loc[index]=[beta_1_param,num_neurons,dropout_rate,num_layers] + result_list + ce_list
    
    # Update the index for the next loop
    index+=1

# After going through all chosen combinations save the csv file for future analysis
df_grid_search.to_csv('df_grid_search.csv')
