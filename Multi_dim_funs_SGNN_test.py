# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
import MV_funs_zoo as mf
import SGNN

# functions to be approximated 
f_arr = [mf.squaredModulus, mf.secondDegreePoly, mf.expSqureSum, mf.expoSinSum, mf.polySinSum,
            mf.invExpSqureSum, mf.sigmoidal, mf.gaussian, mf.linear, mf.constant, mf.gaussian2, 
            mf.MexicanStrawHat, mf.Wave, mf.dumbbell, mf.discRamp, mf.CamassaHolmSol, mf.coshWave]

f = f_arr[15] # change the indext to select a different function in MV_funs_zoo
LAYER_SIGMA_MODE =["identical", "distinct"]
my_sigma_mode = LAYER_SIGMA_MODE[1] # you can select using identfical or distinct sigma for each layer

domain_lb = [-4.0,-4.0, -4.0] #the lower bounds of a n-dim space
domain_ub = [4.0, 4.0, 4.0] #the upper bounds of a n-dim space
grid_size = [40, 40, 40] #the grid size of each dimension
dim =len(domain_lb)

weights_untrainable_layers=[] # last layer fixed weights
layer_weight_initializers={} # last layer unit weights
center_untrainable_layers=[]
width_untrainable_layers=[]

assert (len(domain_lb) == len (domain_ub) and len (grid_size) == len(domain_lb)) # lb, ub, and N should have same dimensions


#generate the training set
training_size = 4096 
mini_batch_size = 256

training_input = tf.constant(np.random.uniform(domain_lb, domain_ub, [training_size,dim]).astype('f'))

training_output = tf.map_fn(f, training_input)

#create the NN model
mu_grid, sigma_arr = SGNN.cmptGaussCenterandWidth(domain_lb,domain_ub,grid_size, sigma_mode=my_sigma_mode) # add scaling factor 

inputs = tf.keras.layers.Input(shape=(dim,))
my_model = SGNN.GaussianNet(mu_grid, sigma_arr, 
                                    1, #output layer neuron number,
                                    weight_initializers=layer_weight_initializers,
                                    weights_untrainable_layers=weights_untrainable_layers, 
                                    center_untrainable_layers=center_untrainable_layers, 
                                    width_untrainable_layers=width_untrainable_layers)
outputs = my_model(inputs)

model = tf.keras.models.Model(inputs, outputs)

print(my_model.summary())
print(model.summary())

#set up the loss function
predictions = model(training_input).numpy()
loss_fn = tf.keras.losses.MeanSquaredError()
loss = loss_fn(training_output, predictions).numpy()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#configureF the model
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['mean_squared_error'])

from timeit import default_timer as timer

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10), cb]

# %%
#training 
history = model.fit(training_input, training_output, batch_size=mini_batch_size, validation_split=0.20, epochs=10000, callbacks=callback)

print("Total computational time", sum(cb.logs), " Seconds.")

#%%
# validation, only use the first two dimensions

mesh_arr = [tf.linspace(lb,ub, N)  for (lb, ub, N) in zip(domain_lb[0:2], domain_ub[0:2], [i * 2 for i in grid_size[0:2]])]
test_grid = tf.meshgrid(*mesh_arr)
test_grid_coord = [tf.reshape(x,(-1,1)) for x in test_grid]
validation_data = tf.squeeze(tf.stack(test_grid_coord, axis=-1))

padding_data = tf.ones([validation_data.shape[0],len(domain_lb)-2])

validation_data = tf.concat([validation_data, padding_data],axis=1)
test_exact_output = tf.map_fn(f, validation_data)
z = tf.reshape(test_exact_output, test_grid[0].shape)
pred_output = model.predict(validation_data)
pred = pred_output.reshape(test_grid[0].shape)


#plotting
# %%
is_plotting  = True

if is_plotting:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = 14

    colors = sns.color_palette("Paired", 7)

    fig = plt.figure(1, figsize=(12,9))
    gs = gridspec.GridSpec(9, 6)
    gs.update (wspace = 10.0, hspace = 10.0)

    ax1 = fig.add_subplot(gs[0:3, 0:3])
    ax1.plot(history.history['loss'], color='black')
    ax1.plot(history.history['val_loss'], color='red', linestyle="--")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2 = fig.add_subplot(gs[0:6, 3:6], projection='3d')
    ax2.plot_surface(test_grid[0].numpy(), test_grid[1].numpy(), z.numpy(), color=colors[1])
    ax2.plot_surface(test_grid[0].numpy(), test_grid[1].numpy(), pred, color=colors[5])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel(r'$f(x, y)$')

    #absolute error plot
    ax3 = fig.add_subplot(gs[3:6, 0:3])
    subfig3 = ax3.contourf( test_grid[0].numpy(), test_grid[1].numpy(), np.abs(pred-z.numpy()))
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.yaxis.set_label_coords(-0.15,0.5)
    cbar3 = fig.colorbar(subfig3) # Add a colorbar to a plot
    cbar3.set_label('Absolute Error')

    #2d projection, prediction

    ax5 = fig.add_subplot(gs[6:9, 0:3])
    cm = plt.cm.get_cmap('RdYlBu')

    subfig5 = ax5.contourf(test_grid[0].numpy(), test_grid[1].numpy(), pred)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    #ax5.yaxis.set_label_coords(-0.15,0.5)
    cbar5 = fig.colorbar(subfig5) # Add a colorbar to a plot
    cbar5.set_label('prediction') 
 
    #2d projection, exact
    ax6 = fig.add_subplot(gs[6:9, 3:6])
    cm = plt.cm.get_cmap('RdYlBu')

    subfig6 = ax6.contourf(test_grid[0].numpy(), test_grid[1].numpy(), z.numpy())
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    cbar6 = fig.colorbar(subfig6) # Add a colorbar to a plot
    cbar6.set_label('Exact') 

    plt.show()
