# This file contains the code for the paper:
# "Accelerating spiking neural network simulations in NumPy or PymoNNto."
# (Vieth et al 2023)
#
# All tests in the paper were conducted on a
# Dell XPS 15 9500 with an Intel i7 10750H
# running windows 11,
# Python 3.11
# with NumPy 1.24.1
# and PymoNNto 3.0.0

import numpy as np
import time

#heating up the CPU
#for i in range(1000):
#    temp = np.random.rand(10000, 5000)*np.random.rand(10000, 5000)+np.random.rand(10000, 5000)


#for i in range(60):
for i in range(1):
    measurements = []

    steps = 1000

    ###########################################
    print('Initialization...')
    ###########################################

    s = np.random.rand(5000) < 0.01    # 1% spikes
    d = np.random.rand(10000) < 0.01   # 1% spikes
    W1 = np.random.rand(10000, 5000).astype(np.float64) # dense DxS synapses
    W2 = np.random.rand(5000, 10000).astype(np.float64) # dense SxD synapses


    ###########################################
    print('\nSynapse Operation...')
    ###########################################

    start = time.time()
    for i in range(steps):
        W1.dot(s)
    t1 = (time.time()-start)/steps*1000
    print('  W1.dot(s):', t1, 'ms')
    measurements.append(t1)


    start = time.time()
    for i in range(steps):
        np.sum(W1[:, s], axis=1)
    t2 = (time.time()-start)/steps*1000
    print('  np.sum(W1[:, s], axis=1):', t2, 'ms', t1/t2, 'x faster')
    measurements.append(t2)


    # same but with W2 (SxD) instead of W1 (DxS):


    start = time.time()
    for i in range(steps):
        W2.T.dot(s)
    t3 = (time.time()-start)/steps*1000
    print('  W2.T.dot(s):', t3, 'ms', t1/t3, 'x faster')
    measurements.append(t3)


    start = time.time()
    for i in range(steps):
        np.sum(W2[s], axis=0)
    t4 = (time.time()-start)/steps*1000
    print('  np.sum(W2[s], axis=0):', t4, 'ms', t1/t4, 'x faster')
    measurements.append(t4)


    ###########################################
    print('\nSTDP...')
    ###########################################

    start = time.time()
    for i in range(steps):
        W1 += d[:, None] * s[None, :]
    t1 = (time.time()-start)/steps*1000
    print('  W1 += d[:, None] * s[None, :]:', t1, 'ms')
    measurements.append(t1)


    #W1[d, s] += 1 # ERROR!


    start = time.time()
    for i in range(steps):
        W1[d[:, None] * s[None, :]] += 1
    t2 = (time.time()-start)/steps*1000
    print('  W1[d[:, None] * s[None, :]] += 1:', t2, 'ms', t1/t2, 'x faster')
    measurements.append(t2)


    start = time.time()
    for i in range(steps):
        W1[np.ix_(d, s)] += 1
    t3 = (time.time()-start)/steps*1000
    print('  W1[np.ix_(d, s)] += 1:', t3, 'ms', t1/t3, 'x faster')
    measurements.append(t3)

    #same but with W2 (SxD) instead of W1 (DxS):

    start = time.time()
    for i in range(steps):
        W2 += s[:, None] * d[None, :]
    t1 = (time.time()-start)/steps*1000
    print('  W2 += s[:, None] * d[None, :]:', t1, 'ms')
    measurements.append(t1)


    #W1[d, s] += 1 # ERROR!


    start = time.time()
    for i in range(steps):
        W2[s[:, None] * d[None, :]] += 1
    t2 = (time.time()-start)/steps*1000
    print('  W2[s[:, None] * d[None, :]] += 1:', t2, 'ms', t1/t2, 'x faster')
    measurements.append(t2)


    start = time.time()
    for i in range(steps):
        W2[np.ix_(s, d)] += 1
    t3 = (time.time()-start)/steps*1000
    print('  W2[np.ix_(s, d)] += 1:', t3, 'ms', t1/t3, 'x faster')
    measurements.append(t3)


    ###########################################
    print('\nAdvanced STDP...')
    ###########################################

    '''
    To create more complex STDP functions (Figure 2b), we can follow the same approach and include two buffers (lists of vectors) 
    that hold the spike history of the source and destination groups (Bs and Bd). 
    Here, the 0 index denotes the most recent spikes:
    '''

    #Version B (multiple blocks)
    Bd = [np.random.rand(10000) < 0.01 for _ in range(2)]
    Bs = [np.random.rand(5000) < 0.01 for _ in range(3)]

    W1[np.ix_(Bd[1], Bs[0])] -= 0.4
    W1[np.ix_(Bd[0], Bs[0])] += 0.6
    W1[np.ix_(Bd[0], Bs[1])] += 1.0
    W1[np.ix_(Bd[0], Bs[2])] += 0.2

    '''
    The method depicted in Figure 2c is also possible, but it requires additional decaying trace variables 
    for both the source and destination groups. 
    These trace variables indirectly store the history of the most recent spiking activity of these groups.
    '''

    #Version C (traces)
    sTrace = np.zeros(5000)
    dTrace = np.zeros(10000)

    sTrace = (sTrace + s) * 0.9
    dTrace = (dTrace + d) * 0.9
    stMask = sTrace>0.01
    dtMask = dTrace>0.01
    W1[np.ix_(d, stMask)] += sTrace[None, stMask]
    W1[np.ix_(dtMask, s)] -= dTrace[dtMask, None]

    '''
    Note that the trace variables have to be converted to binary masks for indexing. 
    To improve the performance the masks have to be as sparse as possible, 
    hence it is necessary to cut of the trace at some point if it gets too small (here 0.01). 
    '''


    ###########################################
    #print('\nClipping...')
    ###########################################

    mask = np.ix_(d, s)
    W1[mask] += 1
    W1[mask] = np.clip(W1[mask], 0.1, 10.0)


    ###########################################
    #print('\nNormalization...')
    ###########################################

    iteration = 100

    if iteration % 100 == 0:
        W1 /= np.sum(W1, axis=1)[:, None] # afferent
        W1 /= np.sum(W1, axis=0)          # efferent


    '''
    One way to accelerate the nomrmalization is to create a variable that tracks the sum of the rows or columns 
    so that the summation operation need not be computed every time. 
    Another approach is to apply indexing to normalize only the required rows and columns:
    '''

    # initialization
    eff_sum = np.sum(W1, axis=0)

    # STDP sparse update sum
    eff_sum[s] += np.sum(W1[:, s], axis=0)

    # Norm
    mask = eff_sum > 1
    W1[:, mask] /= eff_sum[mask]  # efferent norm
    eff_sum[mask].fill(1)

    '''
    We can apply the same technique to afferent synapses. The major limitation here is that this method only works 
    for either afferent or efferent normalization but not both simultaneously.
    A minor issue is that the variable could potentially drift over time depending on its accuracy. However, 
    this can be mitigated by calling the ``initialization'' function periodically.
    We are using the DxS synapse matrix here. For the SxD version, we need to swap the afferent and efferent operations.
    '''


    ###########################################
    print('\nReset operation...')
    ###########################################

    '''
    When optimizing a network, this synaptic mechanisms are obviously the most promising target. 
    However, there are additional steps to optimize a network simulation.
    If we want to zero some neuron or synapse properties repeatedly, one common way is something like this:
    '''

    steps = 100000
    voltage = np.random.rand(5000)

    start = time.time()
    for i in range(steps):
        voltage = voltage * 0.0
    t1 = (time.time()-start)/steps*1000
    print('  voltage = voltage * 0.0:', t1, 'ms')
    measurements.append(t1)

    '''
    However, this approach involves multiplication, which can be expensive and unnecessary for the given task. 
    A better alternative is to create a new variable:
    '''

    start = time.time()
    for i in range(steps):
        voltage = np.zeros(5000)
    t2 = (time.time()-start)/steps*1000
    print('  voltage = np.zeros(5000, dtype=dtype):', t2, 'ms', t1/t2, 'x faster')
    measurements.append(t2)

    '''
    This approach is still sub optimal because we have to allocate new memory during each iteration. 
    The best method here is to use the existing memory section and overwrite it with a new value:
    '''

    start = time.time()
    for i in range(steps):
        voltage.fill(0)
    t3 = (time.time()-start)/steps*1000
    print('  voltage.fill(0):', t3, 'ms', t1/t3, 'x faster')
    measurements.append(t3)

    '''
    In the previous normalization code example, we also observed how the .fill() 
    function can be combined with a masked operation, 
    which can further accelerate the process if the filling is sparse enough or if we already have a pre-computed mask.
    '''

    ###########################################
    print('\nDatatypes...')
    ###########################################
    steps = 1000

    W2 = W2.astype(np.float64)
    start = time.time()
    for i in range(steps):
        #W1[np.ix_(d, s)] += 1
        np.sum(W2[s], axis=0)
    t1 = (time.time()-start)/steps*1000
    print('  float64:', t1, 'ms')
    measurements.append(t1)


    W2 = W2.astype(np.float32)
    start = time.time()
    for i in range(steps):
        #W1[np.ix_(d, s)] += 1
        np.sum(W2[s], axis=0)
    t2 = (time.time()-start)/steps*1000
    print('  float32:', t2, 'ms', t1/t2, 'x faster')
    measurements.append(t2)


    #can create a overflow sometimes because the synapses are relatively big.
    W2 = W2.astype(np.float16)
    start = time.time()
    for i in range(steps):
        #W1[np.ix_(d, s)] += 1
        np.sum(W2[s], axis=0)
    t3 = (time.time()-start)/steps*1000
    print('  float16:', t3, 'ms', t1/t3, 'x faster')
    measurements.append(t3)



    ###########################################
    print('\nPymoNNto naive implementation:')
    ###########################################


    from PymoNNto import *
    settings = {'dtype': float64, 'synapse_mode': DxS}

    class SpikeGeneration(Behavior):
        def initialize(self, neurons):
            neurons.spikes = neurons.vector('bool')
            neurons.spikesOld = neurons.vector('bool')
            neurons.voltage = neurons.vector()
            self.threshold = self.parameter('threshold')

        def iteration(self, neurons):
            neurons.spikesOld = neurons.spikes.copy()
            neurons.spikes = neurons.voltage > self.threshold
            neurons.voltage *= 0.0

    class Input(Behavior):
        def initialize(self, neurons):
            self.strength = self.parameter('strength')
            for s in neurons.synapses(afferent, 'GLU'):
                s.W = s.matrix('random')

        def iteration(self, neurons):
            neurons.voltage += neurons.vector('random')
            for s in neurons.synapses(afferent, 'GLU'):
                input = s.W.dot(s.src.spikes)
                s.dst.voltage += input * self.strength


    class STDP(Behavior):
        def initialize(self, neurons):
            self.speed = self.parameter('speed')

        def iteration(self, neurons):
            for s in neurons.synapses(afferent, 'GLU'):
                s.W += s.dst.spikes[:, None] * s.src.spikesOld[None, :] * self.speed
                s.W = np.clip(s.W, 0.0, 1.0)


    class Norm(Behavior):
        def iteration(self, neurons):
            for s in neurons.synapses(afferent, 'GLU'):
                s.W /= np.sum(s.W, axis=0)


    net = Network(settings=settings)
    NeuronGroup(net, tag='NG', size=7500, behavior={
        1: SpikeGeneration(threshold=0.995),
        2: Input(strength=0.3),
        3: STDP(speed=0.01),
        4: Norm()
    })

    SynapseGroup(net, src='NG', dst='NG', tag='GLU')
    net.initialize()

    measurements.append(net.simulate_iterations(100))




    ###########################################
    print('\nPymoNNto fast implementation:')
    ###########################################


    from PymoNNto import *
    settings = {'dtype': float32, 'synapse_mode': SxD}

    class SpikeGeneration(Behavior):
        def initialize(self, neurons):
            neurons.spikes = neurons.vector('bool')
            neurons.spikesOld = neurons.vector('bool')
            neurons.voltage = neurons.vector()
            self.threshold = self.parameter('threshold')

        def iteration(self, neurons):
            neurons.spikesOld = neurons.spikes.copy()
            neurons.spikes = neurons.voltage > self.threshold
            neurons.voltage.fill(0.0)

    class Input(Behavior):
        def initialize(self, neurons):
            self.strength = self.parameter('strength')
            for s in neurons.synapses(afferent, 'GLU'):
                s.W = s.matrix('random')

        def iteration(self, neurons):
            neurons.voltage += neurons.vector('random')
            for s in neurons.synapses(afferent, 'GLU'):
                input = np.sum(s.W[s.src.spikes], axis=0)
                s.dst.voltage += input * self.strength


    class STDP(Behavior):
        def initialize(self, neurons):
            self.speed = self.parameter('speed')

        def iteration(self, neurons):
            for s in neurons.synapses(afferent, 'GLU'):
                mask = np.ix_(s.src.spikesOld, s.dst.spikes)
                s.W[mask] += self.speed
                s.W[mask] = np.clip(s.W[mask], 0.0, 1.0)


    class Norm(Behavior):
        def iteration(self, neurons):
            if neurons.iteration==1 or neurons.iteration%100 == 0:
                for s in neurons.synapses(afferent, 'GLU'):
                    s.W /= np.sum(s.W, axis=0)


    net = Network(settings=settings)
    NeuronGroup(net, tag='NG', size=7500, behavior={
        1: SpikeGeneration(threshold=0.995),
        2: Input(strength=0.3),
        3: STDP(speed=0.01),
        4: Norm()
    })

    SynapseGroup(net, src='NG', dst='NG', tag='GLU')
    net.initialize()

    measurements.append(net.simulate_iterations(100))

    print(measurements)


    #from PymoNNto.Exploration.Network_UI import *
    #Network_UI(net, modules=get_default_UI_modules(['spikes', 'voltage'])).show()










###########################################
print('\nSparse matrix implementation:')
###########################################

import scipy.sparse as sp

s = np.random.rand(5000) < 0.01  # 1% spikes
d = np.random.rand(10000) < 0.01  # 1% spikes
W1 = np.random.rand(10000, 5000).astype(np.float64)  # dense DxS synapses
W2 = np.random.rand(5000, 10000).astype(np.float64)  # dense SxD synapses

steps = 1000

start = time.time()
for i in range(steps):
    W1[np.ix_(d, s)] += 1.0
t = (time.time() - start) / steps * 1000
print('  W1[np.ix_(d, s)] += 1:', t, 'ms')


W = sp.csc_matrix(W1)
N = W.shape[1]
col = np.repeat(np.arange(N), np.diff(W.indptr))
row = W.indices
start = time.time()
for i in range(steps):
    data = W.data
    data += (d[row] * s[col])
t0 = (time.time() - start) / steps * 1000
print(' sparse mat 100%:', t0, 'ms', t0/t, 'x slower')

#################### 1% density
W1[W1 > 0.01] = 0.0


start = time.time()
for i in range(steps):
    W1[np.ix_(d, s)] += 1.0
t0 = (time.time() - start) / steps * 1000
print(' W1[np.ix_(d, s)] += 1.0  1%:', t0, 'ms')


W = sp.csc_matrix(W1)
W.eliminate_zeros()
N = W.shape[1]
col = np.repeat(np.arange(N), np.diff(W.indptr))
row = W.indices
start = time.time()
for i in range(steps):
    data = W.data
    data += (d[row] * s[col])
t0 = (time.time() - start) / steps * 1000
print(' sparse mat 1%:', t0, 'ms', t0/t, 'x slower')



'''


class STDP2(Behavior):
    def initialize(self, neurons):
        neurons.Trace = neurons.vector()

    def iteration(self, neurons):
        neurons.tMask = neurons.Trace > 0.01

        for s in neurons.synapses(afferent, 'GLU'):

            s.W[np.ix_(s.dst.spikes, s.src.tMask)] += s.src.Trace[None, s.src.tMask]#1#s.dst.Trace[s.dst.tMask]
            s.W[np.ix_(s.dst.tMask, s.src.spikes)] -= s.dst.Trace[s.dst.tMask, None]#1#s.src.Trace[s.src.tMask]

        neurons.Trace = (neurons.Trace + neurons.spikes) * 0.9

        # mask = np.ix_(s.src.spikesOld, s.dst.spikes)
        # s.W[mask] += self.speed
        # s.W[mask] = np.clip(s.W[mask], 0.0, 1.0)


'''