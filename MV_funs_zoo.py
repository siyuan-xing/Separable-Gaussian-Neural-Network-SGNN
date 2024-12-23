import tensorflow as tf

#fun0
def squaredModulus(y): 

    return tf.norm(y)
#fun1
def secondDegreePoly(x):
    x_shift = tf.roll(x, shift=1, axis=0)
    return 0.02*tf.reduce_sum(x*x*x_shift)
#fun2
def expSqureSum(x):
    return 0.2 * tf.reduce_sum(tf.exp(0.02*x**2))
#fun3
def expoSinSum(x):
    x_shift = tf.roll(x, shift=1, axis=0)
    return 0.2 * tf.reduce_sum(tf.exp(0.02*x**2)*tf.sin(x_shift))
#fun4
def polySinSum(x):
    x_shift = tf.roll(x, shift=1, axis=0)
    return 0.02 * tf.reduce_sum(x**2*tf.cos(tf.linspace(1.0, float(len(x)), len(x))*x_shift))
#fun5
def invExpSqureSum(x):

    return 10.0/tf.reduce_sum(tf.exp(0.04*x**2))
#fun6
def sigmoidal(x):

    return 10.0/(1 + tf.exp(-0.2 * tf.reduce_sum(x)))
#fun7
def gaussian(x):

    return 10.0*tf.exp(-0.01*tf.reduce_sum(x**2))
#fun8
def linear(x):

    return tf.reduce_sum(tf.linspace(1.0, len(x), len(x))*x)
#fun9
def constant(x):
    return 1.0

#fun10
def gaussian2(x):
    return 10.0*tf.exp(-4.0*tf.reduce_sum((x+1.0)**2)) + 10.0*tf.exp(-4.0*tf.reduce_sum((x-1.0)**2))
    #return 10.0*tf.exp(-10.0*tf.reduce_sum((x+3.0)**2))

#fun 11
def MexicanStrawHat(x):
    return 10.0*tf.sin(tf.norm(x))/(tf.norm(x))

#fun 12

def Wave(x):
    c=tf.constant(10.0)
    t=x[-1]
    X=x[0:-1]
    return 10.0*tf.sin(tf.norm(X-c*t))

#fun 13

def CamassaHolmSol(X):
    x=X[0]
    return 10.0*(tf.exp(-tf.abs(x-2.0))+tf.exp(-tf.abs(x+2.0)))

#fun 14
def dumbbell(X):
    x=X[0]
    y=X[1]
    z=X[2]
    return (16*y**2-x**4*(4-(x+z)**2))

#fun 15
def discRamp(X): #1D
    x=X[0]
    if x<0:
        return 1+x
    else:
        return -1-x

#fun 16
def coshWave(X):
    r= - 1.5
    x=X[0]
    return (tf.cosh(r*(x-1)))**(-1/r)


if __name__ == '__main__':
    x = tf.constant([1.0,3.0, 4.0])
    print(squaredModulus(x))
    print(secondDegreePoly(x))
    print(expSqureSum(x))
    print(expoSinSum(x))
    print(polySinSum(x))
    print(invExpSqureSum(x))
    print(sigmoidal(x))
    print(gaussian(x))
    print(linear(x))