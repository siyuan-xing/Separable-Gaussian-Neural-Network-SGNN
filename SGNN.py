"""GaussianNN

This module allows the user to create a separable Gaussian neural network.
    
    Author: Siyuan (Simon) Xing
    Email: sixing@calpoly.edu
    Licence: MIT Licence
    Copyright (c) 2024
    version: 1.0.7 (support tensorflow 2.17.0)
    Version: 1.0.6  (add offset to sigma to avoid blowup of gradient when taking derivative of a small sigma
    Version: 1.0.5 (split the trainable configuration for mean and)
"""

import tensorflow as tf
import numpy as np

class GaussianNet(tf.keras.Model):
    """Separable Gaussian neural networks.
        A NN class for separable-Gaussian NNs. The input data is splitted by its dimensions and fed sequentially to each layer.

        Attributes -
            _hidden_layer_num: number of hidden layers. 
            _hidden_layers: an array of hidden_layers.
            _output_layer:  the output layer.
        
        Examples -
            1. Random trainable weights 
                my_model = GaussianNN.GaussianNet(mu_grid, sigma_arr)

                    By default,  random weight initializers are used; all variables (mean, variance, weight) are trainable. 
                
            2.  Init with some untrainable layers
                my_model = GaussianNN.GaussianNet(mu_grid, sigma_arr, 
                                    weight_initializers=layer_weight_initializers,
                                    weights_untrainable_layers=[1], 
                                    center_untrainable_layers=[1], 
                                    width_untrainable_layers=[1])
                where 1 is the layer index (starts from 1).
            
    """
    def __init__(self, mu_arr_list:list, sigma_arr_list:list, output_layer_neuron_num=1,
                 weight_initializers:list=[], weights_untrainable_layers:list=[], 
                 center_untrainable_layers:list=[], width_untrainable_layers:list=[], 
                 data_type='float32', name=None):
        """Constructor for the GaussianNet class. The indexes of the layers start from 1.
        Args:
            mu_arr_list (list): The list of mean arrays for all layers.
            sigma_arr_list (list): The list of variance arrays for all layers.
            output_layer_neuron_num (int): The number of neurons in the output layer. Default is 1.
            weight_initializers (dict): The dictionary of layer initializers. Specify the layers with initializers other than random initializers.
            weights_untrainable_layers (list): The list of layers whose weights are untrainable. The layer index starts from 1.
            center_untrainable_layers (list): The list of layers whose mean is untrainable. The layer index starts from 1.
            width_untrainable_layers (list): The list of layers whose variance is untrainable. The layer index starts from 1. 
            data_type (str): The data type of the network. Default is 'float32'.
            name (str): The name of the network. Default is None.
        """ 
        super(GaussianNet, self).__init__(name=name)

        if len(mu_arr_list) != len(sigma_arr_list):
            raise Exception('The length of mu and sigma arrays have to be identical.')
       
        self.data_type = data_type
        self._hidden_layer_num = len(mu_arr_list)  # layer numbers
        # Pre-processing
        hidden_layer_mean_trainability, hidden_layer_variance_trainability = self.getHiddenLayerCenterandWidthTrainablility(center_untrainable_layers, width_untrainable_layers)

        weight_trainability, my_weight_initializers = self.getHiddenLayerWeightsTrainabilityAndInitializer(weights_untrainable_layers, weight_initializers)

        # Create hidden and output layers
        self._hidden_layers = self.createHiddenLayers(mu_arr_list, sigma_arr_list,
                                        weight_trainability,
                                        hidden_layer_mean_trainability, 
                                        hidden_layer_variance_trainability, 
                                        my_weight_initializers, data_type)        
        
        #output layer is trainable for one dimension input; otherwise, it is untrainable.
        is_output_layer_trainable = False
        if len(mu_arr_list) == 1:
            is_output_layer_trainable = True
        
        #the weights of the output layer are one.
        self._output_layer = tf.keras.layers.Dense(output_layer_neuron_num, kernel_initializer=tf.initializers.ones(), use_bias=False, trainable=is_output_layer_trainable) 

    def call(self, inputs):
        inputs = tf.cast(inputs, self.data_type)
        splitted_data = tf.split(inputs, self._hidden_layer_num, axis=-1)
        
        xn = self._hidden_layers[0](splitted_data[0])
        for i in range(1, self._hidden_layer_num):
            xn = self._hidden_layers[i]([xn, splitted_data[i]])
        outputs = self._output_layer(xn) 

        return outputs

    #utilities
    def createHiddenLayers(self, mu_arr_list, sigma_arr_list, weight_trainibility_arr, mean_trainibility_arr, variance_trainibility_arr, weight_initializer_list, dtype):
        layers = []

        layers.append(FirstGaussian(mu_arr_list[0], sigma_arr_list[0],
                                        mean_isTrainable = mean_trainibility_arr[0], 
                                        variance_isTrainable = variance_trainibility_arr[0], 
                                        data_type=dtype))

        for i in range(1, self._hidden_layer_num ):
            layers.append(Gaussian(len(mu_arr_list[i-1]), mu_arr_list[i], sigma_arr_list[i], 
                                   w_init = weight_initializer_list[i], 
                                   mean_isTrainable = mean_trainibility_arr[i], 
                                   variance_isTrainable = variance_trainibility_arr[i], 
                                   weight_trainable = weight_trainibility_arr[i], 
                                   data_type=dtype)) 

        return layers

    def getHiddenLayerCenterandWidthTrainablility(self, center_untrainable_layers, width_untrainable_layers):
        """
        Determines the trainability of the mean and sigma layers in the hidden layers of the SGNN model.
        By default, Center and Width are all trainable. The user can specify the untrainable layers by providing the indices.

        Args:
            center_untrainable_layers (list): A list of indices representing the mean layers that should be untrainable.
            width_untrainable_layers (list): A list of indices representing the sigma layers that should be untrainable.

        Returns:
            tuple: A tuple containing two lists. The first list represents the trainability of the mean layers, where each
            element corresponds to a hidden layer index. The second list represents the trainability of the sigma layers,
            where each element corresponds to a hidden layer index.
        """
        mean_trainability = [True] * self._hidden_layer_num 
        # Set mean layer trainability by index
        for untrain_mean_idx in center_untrainable_layers:
            mean_trainability[untrain_mean_idx-1] = False
                
        sigma_trainability = [True] * self._hidden_layer_num 
        # Set sigma layer trainability by index
        for untrain_variance_idx in width_untrainable_layers:
            sigma_trainability[untrain_variance_idx-1] = False

        return mean_trainability, sigma_trainability
      
    def getHiddenLayerWeightsTrainabilityAndInitializer(self, weights_untrainable_layers,  initializerDic):
        """Determines the trainability of the weights in the hidden layers of the SGNN model.
        By default, all weights are trainable, with random initializers. The user can specify the untrainable layers by providing the indices.
        Likewise, the user can specify the initializers for the weights in the hidden layers.
        """
        # the index in the dict starts from 1, here we need to convert it to 0 
        total_layer_num = self._hidden_layer_num 
        weight_trainibility = [True] * (total_layer_num)
        weight_trainibility[0] = False # redundant, as the first layer has no weights
        for idx in weights_untrainable_layers:
            weight_trainibility[idx-1] = False

        weights_initializer = [tf.initializers.random_normal()] * (total_layer_num)
        #by default, last dense layer unit init weights, other layers random init weights.

        #update user input 
        if initializerDic != []:
            for (key, value) in initializerDic.items():
                weights_initializer[key-1] = value

        return weight_trainibility, weights_initializer


class FirstGaussian(tf.keras.layers.Layer):
    """The first Gaussian-radial-basis layers, which only accept one input.

    Attributes:
        mu_arr: the array of expected values.
        sigma_arr: the array of variance.

    Example:
        1. Trainable centers and widths
            gaussian_layer = FirstGaussian([1,2,3], [1,1,1]) 
        2. untrainable centers and widths
            gaussian_layer = FirstGaussian([1,2,3], [1,1,1], False, False)
        3. data types other than float32
            gaussian_layer = FirstGaussian([1,2,3], [1,1,1], 'float64') 


    """
    def __init__(self, mu, sigma, mean_isTrainable=True, 
                 variance_isTrainable=True, offset=0.05, 
                 data_type='float32', name=None):
        """Constructor.
        The first layer does not have weights. 
        Args:
            mu_arr: the array of expected values.
            sigma_arr: the array of variance.
            mean_isTrainable: True if centers are trainable.
            variance_isTrainable: True if widths are trainable.
        """
        super(FirstGaussian, self).__init__(name=name)
        self.data_type = data_type
        self.mu = self.add_weight(
            shape=mu.shape,  # Use the shape of the tensor
            initializer=lambda shape, dtype: mu,  # Return the tensor directly
            trainable=mean_isTrainable,
            dtype=data_type,
            name='mu'
        )

        self.sigma = self.add_weight(
            shape=sigma.shape,  # Use the shape of the tensor
            initializer=lambda shape, dtype: sigma,  # Return the tensor directly
            trainable=variance_isTrainable,
            dtype=data_type,
            name='sigma'
        )        
        self.offset = tf.constant(offset, dtype=data_type)

    def call(self, inputs):
        """Forward-pass action for the first gaussian layer.

        Args:
            inputs: N-dimensional spatial point
        Return:
            output of the Gaussian layer.
        """
        inputs=tf.cast(inputs, self.data_type)
        return tf.exp(tf.constant(-0.5, dtype=self.data_type)*(inputs - self.mu)**2/((self.offset+tf.abs(self.sigma))**2))
         

class Gaussian(tf.keras.layers.Layer):
    """Customized layers. This is for layers other than the first layer.
     The layer is composed of 1-D Gaussian-radial-basis functions used for a coordinate
     component dimension.

    Attributes:
        mu_arr: the array of mean values.
        sigma_arr: the array of variance.
        w: weights.

    Example:
    """
    def __init__(self, input_dim, mu, sigma, 
                 w_init = tf.random_normal_initializer(), 
                 mean_isTrainable=True, variance_isTrainable=True, 
                 weight_trainable=True, offset=0.05, 
                 data_type='float32',name=None):
        """Constructor.

        Args:
            input_dim: dimension of input.
            mu: the array of expected values.
            sigma: the array of variance.
            w_init: initial weights.
            mean_isTrainable: True if mean is trainable.
            variance_isTrainable: True is variance is trainable.
            weight_trainable: True if weight is trainable.
        """
        super(Gaussian, self).__init__(name=name)
        self.data_type = data_type
        units = len(mu)
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer=w_init,
            trainable=weight_trainable,
            dtype=data_type,
            name='w'
        )

        self.mu = self.add_weight(
            shape=mu.shape,  # Use the shape of the tensor
            initializer=lambda shape, dtype: mu,  # Return the tensor directly
            trainable=mean_isTrainable,
            dtype=data_type,
            name='mu'
        )

        self.sigma = self.add_weight(
            shape=sigma.shape,  # Use the shape of the tensor
            initializer=lambda shape, dtype: sigma,  # Return the tensor directly
            trainable=variance_isTrainable,
            dtype=data_type,
            name='sigma'
        )

        self.offset = tf.constant(offset, dtype=data_type)
     
    def call(self, inputs):
        """Forward-pass action for the rest gaussian layer.
        Baside the ouput from the previous layer, this layer takes 
        additional input for the corresponding 1-D coordinate component.

        Args:
            inputs: output of the previous layer.
            coord_comp: the coordinate component x in f=e^[-0.5*(x-mu)^2/sigma]. 
        Return:
            output of the Gaussian layer.
        """
        input, coord_comp = inputs
        input=tf.cast(input, self.data_type)
        coord_comp=tf.cast(coord_comp, self.data_type)

        return  tf.reduce_sum(self.w * tf.expand_dims(input,axis=-1), axis=1) * \
                    tf.exp(tf.constant(-0.5,dtype=self.data_type)* \
                    (coord_comp-self.mu)**2/(self.offset+tf.abs(self.sigma))**2)

#Utility
def cmptGaussCenterandWidth(lb_arr:list, ub_arr:list, N_arr:list, sigma_mode="identical", data_type='float32'):
    """Compute the array of initial centers and widths by the lower and upper bounds, and grid_size (N_arr) of N dimensions.
    
    Args:
        lb_arr (list): Array of lower bounds for each dimension.
        ub_arr (list): Array of upper bounds for each dimension.
        N_arr (list): Array of grid sizes for each dimension.
        sigma_mode (str, optional): Mode for determining the width of the Gaussians. 
            "identical" means all Gaussians have the same width (default).
            "distinct" means each Gaussian has a distinct width.
        data_type (str, optional): Data type of the arrays (default is 'float32').
    
    Returns:
        tuple: A tuple containing two lists:
            - mu_arr_list (list): List of arrays representing the initial centers for each dimension.
            - sigma_arr_list (list): List of arrays representing the widths for each dimension.
    """
    mu_arr_list, sigma_arr_list = [], []
   
    for (lb, ub, N) in zip(lb_arr, ub_arr, N_arr):
        lb = tf.cast(lb, dtype=data_type)
        ub = tf.cast(ub, dtype=data_type)
        mu = tf.cast(tf.linspace(lb,ub,N),dtype=data_type) # the lb and ub will be included.
        if sigma_mode == "identical":
            sigma = tf.constant(mu[1]-mu[0],dtype=data_type) #sigma = 1 step all neurons of a layer
        elif sigma_mode =="distinct":
            sigma = (mu[1]-mu[0])*tf.ones(N,dtype=data_type) #sigma = 1 step per neuron of a layer
        else:
            raise Exception('Unknown sigma mode.')

        mu_arr_list.append(mu)
        sigma_arr_list.append(sigma)
    
    return mu_arr_list, sigma_arr_list
