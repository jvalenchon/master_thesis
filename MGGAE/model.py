# based on the Github code of Van den Berg https://github.com/riannevdberg/gc-mc
#modified by Juliette Valenchon

from __future__ import print_function
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.global_step = tf.Variable(0, trainable=False)

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


"""class RecommenderGAE(Model):
    def __init__(self, placeholders, input_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 self_connections=False, **kwargs):
        super(RecommenderGAE, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[2]

        self._rmse()

    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)

        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

        tf.summary.scalar('rmse_score', self.rmse)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=False))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))
        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.hidden[0],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=True))

        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))

class RecommenderGAEDisease(Model):
    def __init__(self, placeholders, input_dim, num_support, learning_rate, num_basis_functions, hidden, num_users, num_items, accum, self_connections=False, **kwargs):
        super(RecommenderGAEDisease, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        #self.class_values = placeholders['class_values']
        self.indice_labels=placeholders['indices_labels']

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()
        #tf.summary.scalar('layer', self.layers[0])
        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[2]



    def frobenius_norm_square(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        return tensor_sum

    def _loss(self):

        self.outputs=tf.reshape(self.outputs, [-1])
        self.indices = tf.where(tf.equal(self.v_indices, self.indice_labels))

        self.classification= tf.gather_nd(self.outputs, self.indices)#self.outputs[indices]
        self.labels_class = tf.gather_nd(self.labels, self.indices)#self.labels[indices]

        self.indices_features = tf.where(tf.not_equal(self.v_indices, self.indice_labels))

        self.reconstr = tf.gather_nd(self.outputs, self.indices_features)#self.outputs[indices_features]
        self.labels_feat = tf.gather_nd(self.labels, self.indices_features)# self.labels[indices_features]

        self.l2_regu = tf.nn.l2_loss(self.weight_dense_u)+tf.nn.l2_loss(self.weight_dense_v)+tf.nn.l2_loss(self.weight)+tf.nn.l2_loss(self.weight_gcn_u)+tf.nn.l2_loss(self.weight_gcn_v)
        self.loss_frob = self.frobenius_norm_square(tf.subtract(self.reconstr,self.labels_feat))/(self.num_items*self.num_users)#tf.shape(self.reconstr)[0]

        #self.output_nn = tf.slice(self.outputs, begin = [0, self.outputs.get_shape().as_list()[1]-1], size = [self.outputs.get_shape().as_list()[0], 1]) #does not work out
        #self.output_nn=tf.sigmoid(self.classification)
        #self.classification=tf.sigmoid(self.classification)
        self.binary_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_class, logits = self.classification)

        self.loss = self.loss_frob +10* self.binary_entropy+0.001*self.l2_regu

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('indices', self.indices)
        tf.summary.scalar('labels feat', self.labels)
        tf.summary.scalar('output feat', self.outputs)
        tf.summary.scalar('label class', self.labels_class)
        tf.summary.scalar('output class', self.classification)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    #def _rmse(self):
    #    self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

    #    tf.summary.scalar('rmse_score', self.rmse)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=False))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=0., #0.5,#self.dropout,
                                        logging=self.logging,
                                        bias=True,
                                        share_user_item_weights=True))
        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.hidden[0],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=0.,# 0.5,#self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=True))

        self.layers.append(BilinearMixtureDisease(num_classes=1, u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=True,
                                           dropout=0.,#0.5,
                                           act=lambda x: x,
                                           num_weights=1,
                                           logging=self.logging,
                                           diagonal=False))
    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model
        self.activations.append(self.inputs)
        # gcn layer
        layer = self.layers[0]
        self.input_gcn = self.inputs
        #gcn_hidden = layer(self.inputs)
        #self.gcn_hidden = gcn_hidden
        gcn_u, gcn_v, x_u, x_v, weight_gcn_u, weight_gcn_v=layer(self.inputs)
        self.input_g_u = x_u
        self.input_g_v = x_v
        self.weight_gcn_u=weight_gcn_u
        self.weight_gcn_v=weight_gcn_v

        #gcn_u = gcn_hidden[0]
        #gcn_v = gcn_hidden[1]
        self.gcn_u, self.gcn_v = gcn_u, gcn_v

        #feat_u = feat_hidden[0]
        #feat_v = feat_hidden[1]

        # dense layer
        layer = self.layers[1]
        dense_gcn = layer([gcn_u, gcn_v])
        self.dense=dense_gcn
        #dense_gcn_v = layer(gcn_v)

        #feat_hidden = layer([self.u_features_side, self.v_features_side])

        # concat dense layer
        layer = self.layers[2]
        input_u = dense_gcn[0]#tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = dense_gcn[1]#tf.concat(values=[gcn_v, feat_v], axis=1)
        self.weight_dense_u = dense_gcn[2]
        self.weight_dense_v = dense_gcn[3]
        self.input_u=input_u
        self.input_v=input_v
        concat_hidden, u_inputs, v_inputs, weight = layer([input_u, input_v])
        self.concat_hidden = concat_hidden
        self.u_inputs=u_inputs
        self.v_inputs=v_inputs
        self.weight=weight

        #self.activations.append(concat_hidden)

        # Build sequential layer model
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.outputs = self.activations[-1]
        self.outputs = self.concat_hidden

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.variables = variables
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

class RecommenderGAEDisease_hyperparam(Model):
    def __init__(self, placeholders, input_dim, num_support, learning_rate, num_basis_functions, hidden, num_users, num_items, accum, gamma, beta, self_connections=False, **kwargs):
        super(RecommenderGAEDisease_hyperparam, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        #self.class_values = placeholders['class_values']
        self.indice_labels=placeholders['indices_labels']

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate
        self.gamma = gamma #param loss function
        self.beta = beta

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()
        #tf.summary.scalar('layer', self.layers[0])
        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[2]

    def frobenius_norm_square(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        return tensor_sum

    def _loss(self):

        self.outputs=tf.reshape(self.outputs, [-1])
        self.indices = tf.where(tf.equal(self.v_indices, self.indice_labels))

        self.classification= tf.gather_nd(self.outputs, self.indices)#self.outputs[indices]
        self.labels_class = tf.gather_nd(self.labels, self.indices)#self.labels[indices]

        self.indices_features = tf.where(tf.not_equal(self.v_indices, self.indice_labels))

        self.reconstr = tf.gather_nd(self.outputs, self.indices_features)#self.outputs[indices_features]
        self.labels_feat = tf.gather_nd(self.labels, self.indices_features)# self.labels[indices_features]

        self.l2_regu = tf.nn.l2_loss(self.weight_dense_u)+tf.nn.l2_loss(self.weight_dense_v)+tf.nn.l2_loss(self.weight)+tf.nn.l2_loss(self.weight_gcn_u)+tf.nn.l2_loss(self.weight_gcn_v)
        self.loss_frob = self.frobenius_norm_square(tf.subtract(self.reconstr,self.labels_feat))/(self.num_items*self.num_users)#tf.shape(self.reconstr)[0]

        #self.output_nn = tf.slice(self.outputs, begin = [0, self.outputs.get_shape().as_list()[1]-1], size = [self.outputs.get_shape().as_list()[0], 1]) #does not work out
        #self.output_nn=tf.sigmoid(self.classification)
        #self.classification=tf.sigmoid(self.classification)
        self.binary_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_class, logits = self.classification)

        self.loss = self.loss_frob +self.gamma* self.binary_entropy+self.beta*self.l2_regu

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('indices', self.indices)
        tf.summary.scalar('labels feat', self.labels)
        tf.summary.scalar('output feat', self.outputs)
        tf.summary.scalar('label class', self.labels_class)
        tf.summary.scalar('output class', self.classification)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    #def _rmse(self):
    #    self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

    #    tf.summary.scalar('rmse_score', self.rmse)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=False))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=0.0,#self.dropout,#0.5, #0.5,#self.dropout,
                                        logging=self.logging,
                                        bias=True,
                                        share_user_item_weights=True))
        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.hidden[0],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=0.0,#self.dropout, #0.5,# 0.5,#self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=True))

        self.layers.append(BilinearMixtureDisease(num_classes=1, u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=True,
                                           dropout=0.0, #self.dropout,#0.5,#0.5,
                                           act=lambda x: x,
                                           num_weights=1,
                                           logging=self.logging,
                                           diagonal=False))
    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model
        self.activations.append(self.inputs)
        # gcn layer
        layer = self.layers[0]
        self.input_gcn = self.inputs
        #gcn_hidden = layer(self.inputs)
        #self.gcn_hidden = gcn_hidden
        gcn_u, gcn_v, x_u, x_v, weight_gcn_u, weight_gcn_v=layer(self.inputs)
        self.input_g_u = x_u
        self.input_g_v = x_v
        self.weight_gcn_u=weight_gcn_u
        self.weight_gcn_v=weight_gcn_v

        #gcn_u = gcn_hidden[0]
        #gcn_v = gcn_hidden[1]
        self.gcn_u, self.gcn_v = gcn_u, gcn_v

        #feat_u = feat_hidden[0]
        #feat_v = feat_hidden[1]

        # dense layer
        layer = self.layers[1]
        dense_gcn = layer([gcn_u, gcn_v])
        self.dense=dense_gcn
        #dense_gcn_v = layer(gcn_v)

        #feat_hidden = layer([self.u_features_side, self.v_features_side])

        # concat dense layer
        layer = self.layers[2]
        input_u = dense_gcn[0]#tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = dense_gcn[1]#tf.concat(values=[gcn_v, feat_v], axis=1)
        self.weight_dense_u = dense_gcn[2]
        self.weight_dense_v = dense_gcn[3]
        self.input_u=input_u
        self.input_v=input_v
        concat_hidden, u_inputs, v_inputs, weight = layer([input_u, input_v])
        self.concat_hidden = concat_hidden
        self.u_inputs=u_inputs
        self.v_inputs=v_inputs
        self.weight=weight

        #self.activations.append(concat_hidden)

        # Build sequential layer model
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.outputs = self.activations[-1]
        self.outputs = self.concat_hidden

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.variables = variables
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)"""

class MG_GAE(Model):
    def __init__(self, placeholders, input_dim, num_support, learning_rate, hidden, num_users, num_items, gamma, beta, **kwargs):
        super(MG_GAE, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.indice_labels=placeholders['indices_labels']

        self.hidden = hidden
        self.num_support = num_support
        self.input_dim = input_dim
        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.gamma = gamma #param loss function
        self.beta = beta

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()
        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[2]

    def frobenius_norm_square(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        return tensor_sum

    def _loss(self):
        self.outputs=tf.reshape(self.outputs, [-1])
        self.indices = tf.where(tf.equal(self.v_indices, self.indice_labels))

        self.classification= tf.gather_nd(self.outputs, self.indices)
        self.labels_class = tf.gather_nd(self.labels, self.indices)

        self.indices_features = tf.where(tf.not_equal(self.v_indices, self.indice_labels))

        self.reconstr = tf.gather_nd(self.outputs, self.indices_features)
        self.labels_feat = tf.gather_nd(self.labels, self.indices_features)

        self.l2_regu = tf.nn.l2_loss(self.weight_dense_u)+tf.nn.l2_loss(self.weight_dense_v)+tf.nn.l2_loss(self.weight)+tf.nn.l2_loss(self.weight_gcn_u)+tf.nn.l2_loss(self.weight_gcn_v)
        self.loss_frob = self.frobenius_norm_square(tf.subtract(self.reconstr,self.labels_feat))/(self.num_items*self.num_users)

        self.binary_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.labels_class, logits = self.classification)

        self.loss = self.loss_frob +self.gamma* self.binary_entropy+self.beta*self.l2_regu

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('indices', self.indices)
        tf.summary.scalar('labels feat', self.labels)
        tf.summary.scalar('output feat', self.outputs)
        tf.summary.scalar('label class', self.labels_class)
        tf.summary.scalar('output class', self.classification)


    def _build(self):
        self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=0.0,
                                        logging=self.logging,
                                        bias=True,
                                        share_user_item_weights=True))

        self.layers.append(Dense(input_dim=self.hidden[0],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=0.0,
                                 logging=self.logging,
                                 share_user_item_weights=True))

        self.layers.append(BilinearMixtureDisease(num_classes=1, u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=True,
                                           dropout=0.0,
                                           act=lambda x: x,
                                           num_weights=1,
                                           logging=self.logging,
                                           diagonal=False))
    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model
        self.activations.append(self.inputs)
        # gcn layer
        layer = self.layers[0]
        self.input_gcn = self.inputs

        gcn_u, gcn_v, x_u, x_v, weight_gcn_u, weight_gcn_v=layer(self.inputs)
        self.input_g_u = x_u
        self.input_g_v = x_v
        self.weight_gcn_u=weight_gcn_u
        self.weight_gcn_v=weight_gcn_v

        self.gcn_u, self.gcn_v = gcn_u, gcn_v


        # dense layer
        layer = self.layers[1]
        dense_gcn = layer([gcn_u, gcn_v])
        self.dense=dense_gcn

        # concat dense layer
        layer = self.layers[2]
        input_u = dense_gcn[0]
        input_v = dense_gcn[1]
        self.weight_dense_u = dense_gcn[2]
        self.weight_dense_v = dense_gcn[3]
        self.input_u=input_u
        self.input_v=input_v
        concat_hidden, u_inputs, v_inputs, weight = layer([input_u, input_v])
        self.concat_hidden = concat_hidden
        self.u_inputs=u_inputs
        self.v_inputs=v_inputs
        self.weight=weight

        # Build sequential layer model
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.outputs = self.activations[-1]
        self.outputs = self.concat_hidden

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.variables = variables
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
