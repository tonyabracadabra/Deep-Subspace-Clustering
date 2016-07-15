from sp import getSparcityPrior
class DeepSubspaceClustering:

    def __init__(self, inputX, C=None, hidden_dims=[300,150,300], lambda1=0.01, lamda2=0.01, activation='tanh', \
                weight_init='uniform', noise=None, lr=0.001, batch_size=100, print_step=50):

        self.print_step = print_step
        self.lr = lr
        self.noise = noise
        self.dims = hidden_dims

        # M must be a even number
        assert len(hidden_dims) % 2 == 1
        # self.depth = len(dims)

        # These are X that be input, not the symbolic variable of tensorflow
        self.inputX = inputX

        if C == None:
            C = getSparcityPrior(inputX)
            self.C = tf.constant(C)

        self.hidden_layers
        self.X = _add_noise(tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='X'))

        input_hidden = self.X
        weights, biases = init_layer_weight(hidden_dims, inputX, weight_init)

        # J3 regularization term
        J3_list = []
        for init_w, init_b in zip(weights, biases):
            self.hidden_layers.append(DenseLayer(input_hidden, init_w, init_b, activation=activation))
            input_hidden = self.hidden_layers[-1].output
            J3_list.append(tf.nn.l2_loss(self.hidden_layers[-1].w))
            J3_list.append(tf.nn.l2_loss(self.hidden_layers[-1].b))

        J3 = lambda2 * tf.add_n(J3_list)

        self.H_M = self.hidden_layers[-1].output
        # H(M/2) the output of the mid layer
        self.H_M_2 = self.hidden_layers[len(hidden_dims)/2].output

        # calculate loss J1
        J1 = tf.nn.l2_loss(tf.sub(self.X, self.H_M))

        # calculate loss J2
        J2 = lambda1 * tf.nn.l2_loss(tf.sub(tf.transpose(self.H_M_2), \
                                     tf.matmul(tf.transpose(self.H_M_2), self.C)))

        self.cost = J1 + J2 + J3


    def train(self, batch_size=100, epochs=100):
    	sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        batch_generator = GenBatch(self.inputX, batch_size)
        n_batch = batch_generator.n_batch

        for i in xrange(epochs):
            # x_batch, y_batch = get_batch(self.X_train, self.y_train, batch_size)
            batch_generator.resetIndex()
            for j in xrange(n_batch+1):
                x_batch, y_batch = batch_generator.get_batch()
                sess.run(self.optimizer, feed_dict={self.X: x_batch})

            self.losses.append(sess.run(self.cost, feed_dict={self.X: x_batch}))

            if i % self.print_step == 0:
                print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        self.result = sess.run(self.H_M_2, feed_dict={self.X: self.inputX})


    def _add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if self.noise == 'mask':
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == None:
            return x