
# Import modules
import numpy as np
import random as re
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from tensorflow.python.ops.gen_nn_ops import relu, elu
from tensorflow.python.ops.nn_ops import leaky_relu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


# Used for splitting data into different proportions
# Shuffling must be done BEFORE calling this definition
def split_data(proportion, data, labels):

    # Find number of instances
    num_examples = data.shape[0]
    # Create index to split on
    split_idx = int(proportion * num_examples)
    
    # Split data into validation and optimization pieces
    data_1, data_2 = data[:split_idx], data[split_idx:]
    # Split labels into validation and optimization pieces
    labels_1, labels_2 = labels[:split_idx], labels[split_idx:]
    
    # Return data as valid_images,valid_labels,opt_images,opt_labels
    return data_1, labels_1, data_2, labels_2

# Once-hot encodes labels
def one_hot_encode(labels):
    # Instantiate one hot encoder
    enc_lab=OneHotEncoder()
    
    # Fit the encoder with its labels
    enc_lab.fit(labels.reshape(-1,1))
    
    return enc_lab.transform(labels.reshape(-1,1)).toarray()

# Used to either return default Adam Optimizer parameters or randomly choose them
def generate_adam_param(default=False):
    
    if default: # Use default values for this optimzer
        return 0.001, 0.9, 0.999
    # Randomly choose parameters
    else:
        # Lists of possible parameters
        lr_list = [0.01, 0.03, 0.001, 0.003, 0.0001, 0.0003]
        beta1_list = [0.9, 0.99, 0.999]
        beta2_list = [0.999, 0.9999, 0.99999]
        # Choose random selection
        lr, b_1, b_2 = re.choice(lr_list), re.choice(beta1_list), re.choice(beta2_list)
        return lr, b_1, b_2
    
# Function to plot learning curve
def plot_learning_curve(acc_list_train,acc_list_val,error_list_train,error_list_val):
    # Define figure
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Accuracy
    ax1.set_title('Model Accuracy')
    ax1.plot(acc_list_train)
    ax1.plot(acc_list_val)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'validation'], loc='upper left')
    
    # Loss
    ax2.set_title('Cross-Entropy')
    ax2.plot(error_list_train)
    ax2.plot(error_list_val)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Cross-Entropy')
    ax2.legend(['train', 'validation'], loc='upper left')

    fig.set_size_inches(20, 10)
    # Save plot to disk
    plt.savefig('learning_curve.png')
    
    #plt.show()
    
# Adds chosen type of noise to images
def noisy(noise_typ,image):
    
    if noise_typ=="gauss":
        mean=0
        var=0.01
        sigma=var**0.5
        
        # Run this if multi-channel image
        if len(image.shape)==3:
            row,col,ch=image.shape
            gauss=np.random.normal(mean,sigma,(row,col,ch))
        # Run this if only grayscale image
        else:
            row,col=image.shape
            gauss=np.random.normal(mean,sigma,(row,col))
        
        noisy=image+gauss
        return noisy
    
    elif noise_typ == "s&p":
        s_vs_p=0.5
        amount=0.01
        out=np.copy(image)
        #Salt mode
        num_salt=np.ceil(amount*image.size*s_vs_p)
        coords=[np.random.randint(0,i-1,int(num_salt)) for i in image.shape]
        out[coords]=1
        #pepper mode
        num_pepper=np.ceil(amount*image.size*(1-s_vs_p))
        coords=[np.random.randint(0,i-1,int(num_pepper)) for i in image.shape]
        out[coords]=0
        return out
    
    elif noise_typ=='poisson':
        vals=len(np.unique(image))
        vals=2**np.ceil(np.log2(vals))
        # Image*vals is lambda, and that must be greater than or equal to zero, hence absolute value
        noisy=np.random.poisson(abs(image*vals))/float(vals)
        return noisy
    
    elif noise_typ=="speckle":
        
        param = 0.1
        # Run this if multi-channel image
        if len(image.shape)==3:
            row,col,ch=image.shape
            gauss=np.random.randn(row,col,ch)
            gauss=gauss.reshape(row,col,ch)
        # Run this if grayscale image
        else:
            row,col=image.shape
            gauss=np.random.randn(row,col)
            gauss=gauss.reshape(row,col)
            
        noisy=image+image*gauss*param
        return noisy
    
# Takes in images and outputs same images with noise added
def transform_image(data,labels):
    # Note we return the exact same labels as we sent in
    '''Parameters
       labels:np array
       data:np nd array
       One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

    '''
    
    stack_data=[]
    # Reshape data from flattened image back into original image
    for i in range(data.shape[0]):
        item=data[i,:].reshape(28,28)
        stack_data.append(item)
        
    available_transformations=['s&p','speckle','poisson','gauss']
    dup_Arrays = np.empty((data.shape[0],data.shape[1]))
    for index,image in enumerate(stack_data):

        # Apply a single transformation or a combination of transformations
        num_trans_to_apply=re.randint(1,len(available_transformations))
        j=0
        
        while j<num_trans_to_apply:
            new_img=noisy(re.choice(available_transformations),image)
            image=new_img
            j+=1
            
        #flatten this new image
        image=image.reshape(1,784)
        
        # Concatenate to original np arrays of data and labels
        dup_Arrays[index,:] = image
        
    return dup_Arrays,labels
