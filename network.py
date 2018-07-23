import tensorflow as tf
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dropout
import CONFIG

class ConvHead():

    def __init__(self):

        # customize the first conv layer with additional input channels
        self.conv1_addi = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')

        resnet50 = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    pooling= 'avg'
                    )

        # extract the first conv layer of res50
        self.conv1_res = resnet50.get_layer(index=1)
        res50_nohead = resnet50.layers[2:]

        # freeze batch_norm layers
        for i in range(len(res50_nohead)):
            if isinstance(res50_nohead[i], BatchNormalization):
                res50_nohead[i].trainable = False
        # extract blocks from res50
        self.res1_block = res50_nohead[0:3]
        self.res2_block = res50_nohead[3:35]
        self.res3_block = res50_nohead[35:77]
        self.res4_block = res50_nohead[77:139]
        #self.res5_block = self.res50_nohead[139:171]
        # with average pooling layer in the end
        self.res5_block = res50_nohead[139:172]



    def build(self, rgb, depth):

        rgb_conv = self.conv1_res(rgb)
        d_conv = self.conv1_addi(depth)
        #p_conv = self.conv1_addi(p_map)
        x1 = self.res1_forward(rgb_conv, d_conv)

        x2 = self.res2_forward(x1)
        x3 = self.res3_forward(x2)
        x4 = self.res4_forward(x3)
        x5_unpool, x5 = self.res5_forward(x4)
        return x1, x2, x3, x4, x5_unpool, x5

    def forward_block(self, block, input):
        x = block[0](input)
        for layer in block[1:]:
            if isinstance(layer, MaxPooling2D):
                x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            else:
                x = layer(x)
        return x

    def identity_block(self, block, input):
        x = self.forward_block(block[0:-2],input)
        x = block[-2]([x, input])
        x = block[-1](x)
        return x

    def conv_block(self, block, input):
        x = self.forward_block(block[0:-5],input)
        y = block[-5](input)
        x = block[-4](x)
        y = block[-3](y)
        x = block[-2]([x, y])
        x = block[-1](x)
        return x

    def res1_forward(self, rgb_conv, d_conv):

        x = rgb_conv + d_conv
        x = self.forward_block(self.res1_block, x)
        #print(x.get_shape().as_list())
        return x

    def res2_forward(self, input):
        x = self.conv_block(self.res2_block[0:12], input)
        x = self.identity_block(self.res2_block[12:22],x)
        x = self.identity_block(self.res2_block[22:], x)
        return x

    def res3_forward(self,input):
        x = self.conv_block(self.res3_block[0:12], input)
        x = self.identity_block(self.res3_block[12:22], x)
        x = self.identity_block(self.res3_block[22:32], x)
        x = self.identity_block(self.res3_block[32:], x)
        return x

    def res4_forward(self, input):
        x = self.conv_block(self.res4_block[0:12], input)
        x = self.identity_block(self.res4_block[12:22], x)
        x = self.identity_block(self.res4_block[22:32], x)
        x = self.identity_block(self.res4_block[32:42], x)
        x = self.identity_block(self.res4_block[42:52], x)
        x = self.identity_block(self.res4_block[52:], x)
        return x

    def res5_forward(self, input):
        x = self.conv_block(self.res5_block[0:12], input)
        x = self.identity_block(self.res5_block[12:22], x)
        x_unpool = self.identity_block(self.res5_block[22:32], x)
        x = self.res5_block[32](x_unpool)
        return x_unpool, x

class FrameEncoder():
    def __init__(self, hidden_nodes, n_unrolls, keep_prob = 0.2):
        self.hidden_nodes = hidden_nodes
        self.n_unrolls = n_unrolls
        self.rnn_layers = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True),
                output_keep_prob=keep_prob)
            for size in hidden_nodes]
        self.multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(self.rnn_layers)
        return

    def build(self, rss, phase, x5, batch_size):

        # concat the rss, phase signals and high-level frame representation
        conv_f = tf.squeeze(x5, [1,2])
        conv_f_shape = conv_f.get_shape().as_list()
        rss_tile = tf.tile(rss, [1, conv_f_shape[-1]])
        phase_tile = tf.tile(phase, [1, conv_f_shape[-1]])
        stack = tf.stack([rss_tile, phase_tile, conv_f], 1)

        #Encode the concatenated feature with 2-layer LSTM, output [batch, 2048]
        x = self.LSTM_Encoder(stack,self.n_unrolls, batch_size)
        return x

    def LSTM_Encoder(self, x, n_unrolls, batch_size):
        # x shape: [None, 3, 2048], the lstm encoder rolls axis 1

        # reshape x to the format [batch_size, n_unrolls, 3, 2048]
        x_shape = x.get_shape().as_list()
        x_reshape = tf.reshape(x, tf.stack([batch_size, n_unrolls, x_shape[-2], x_shape[-1]]))

        # x_batch_reshape [batch_size * 2048, n_unrolls, 3]
        x_transpose = tf.transpose(x_reshape, [3,0,1,2])
        x_batch_reshape = tf.reshape(x_transpose,tf.stack([batch_size*x_shape[-1], n_unrolls, x_shape[-2]]))
        #print(x_batch_reshape.get_shape().as_list())

        # lstm_out [batch_size * 2048, n_unrolls, n_channels]
        with tf.variable_scope("lstm_"):
            lstm_out, lstm_state = tf.nn.dynamic_rnn(self.multi_layer_cell, x_batch_reshape, dtype=tf.float32)
        #print(lstm_out.get_shape().as_list())

        # lstm_out_transpose [batch_size, n_unrolls, n_channels, 2048]
        lstm_out_reshape = tf.reshape(lstm_out, tf.stack([x_shape[-1], batch_size, n_unrolls, self.hidden_nodes[-1]]))
        lstm_out_transpose = tf.transpose(lstm_out_reshape,[1,2,3,0])
        #print(lstm_out_transpose.get_shape().as_list())

        x_out = tf.reshape(lstm_out_transpose,tf.stack([batch_size * n_unrolls, self.hidden_nodes[-1], x_shape[-1]]))
        x_out = tf.reduce_mean(x_out,1)
        #print(x_out.get_shape().as_list())
        """
        x_out_list = []
        with tf.variable_scope("lstm_"):
            for i in range(x_shape[-1]):
                # x_slice_i shape [batch_size, n_unrolls, 3]
                x_slice_i = tf.squeeze(tf.slice(x_reshape, [0,0,0,i], [batch_size,n_unrolls, x_shape[-2], 1]))
                # x_out_i shape [batch_size, n_unrolls, n_hidden]
                x_out_i = tf.nn.dynamic_rnn(self.multi_layer_cell, x_slice_i, dtype=tf.float32)

                x_out_list.append(x_out_i)

        # x_out_stack shape [batch_size, n_unrolls, n_hidden, 2048]
        x_out_stack = tf.stack(x_out_list, -1)
        x_out_stack_shape = x_out_stack.get_shape().as_list()

        # x_out [None, n_hidden, 2048]
        x_out = tf.reshape(x, tf.stack([batch_size*n_unrolls, x_out_stack_shape[-2], x_out_stack_shape[-1]]))
        """
        return x_out

class FrameDecoder():
    def __init__(self):
        return

    def res_block(self,input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + stage + block + '_branch'
        bn_name_base = 'bn' + stage + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(
            filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
            x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_res_block(self, input_tensor, kernel_size, conv_filters, filters, stage, block):
        conv_name_base = 'res' + stage + block + '_branch'
        bn_name_base = 'bn' + stage + block + '_branch'
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(conv_filters, (3, 3), padding='same', name=conv_name_base + 'c')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)
        x = Activation('relu')(x)

        x = self.res_block(x, kernel_size, filters, stage, block)
        return x


    def weighted_channels(self, lstm_feature, x5_unpool):
        # attention mechanism
        # lstm_feature [batch, 2048],  x5_unpool [batch, h, w, 2048]

        # normalize the weight by softmax
        exp_sum = tf.reduce_sum(tf.exp(lstm_feature),1,keepdims=True)
        softmax_lstm_f = tf.exp(lstm_feature) / exp_sum

        # UpSampling the weights to 2D map
        lstm_feature_1D = tf.expand_dims(softmax_lstm_f, axis=1)
        lstm_feature_2D = tf.expand_dims(lstm_feature_1D, axis=1)
        x5_shape = x5_unpool.get_shape().as_list()
        lstm_2D = UpSampling2D(size=(x5_shape[1], x5_shape[2]))(lstm_feature_2D)

        # element-wise multiply
        x5_weighted = tf.multiply(lstm_2D, x5_unpool)
        #print(x5_weighted.get_shape().as_list())
        return x5_weighted

    def build(self, lstm_feature, x5_unpool, x4, x3, x2):
        x5_weighted = self.weighted_channels(lstm_feature, x5_unpool)
        d_x5 = self.conv_res_block(x5_weighted, 3, 128, [128, 128, 128], stage='d5', block='a')


        d_x4_res = self.conv_res_block(x4, 3, 128, [128, 128, 128], stage='d4', block='a')
        d_x5_up = UpSampling2D(size=(2, 2))(d_x5)
        d_add4 = tf.add(d_x4_res, d_x5_up)
        d_x4 = self.res_block(d_add4, 3, [128, 128, 128], stage='d4', block='b')

        d_x3_res = self.conv_res_block(x3, 3, 128, [128, 128, 128], stage='d3', block='a')
        d_x4_up = UpSampling2D(size=(2, 2))(d_x4)
        d_add3 = tf.add(d_x3_res, d_x4_up)
        d_x3 = self.res_block(d_add3, 3, [128, 128, 128], stage='d3', block='b')

        #print(d_x3.get_shape().as_list())

        d_x2_res = self.conv_res_block(x2, 3, 128, [128, 128, 128], stage='d2', block='a')
        d_x3_up = UpSampling2D(size=(2, 2))(d_x3)
        d_add2 = tf.add(d_x2_res, d_x3_up)
        d_x2 = self.res_block(d_add2, 3, [128, 128, 128], stage='d2', block='b')

        # output unnormalized log-probabilities
        x_out_log = Conv2D(2, (3, 3), padding='same', name="cnn_out")(d_x2)
        x_out_log = UpSampling2D(size=(4, 4))(x_out_log)


        """
        x_out_logi = Activation('softmax')(x_out)
        
        # output the prob map
        out_shape = x_out_logi.get_shape().as_list()
        x_ob_prob = tf.slice(x_out_logi, [0, 0, 0, 0], [out_shape[0], out_shape[1], out_shape[2], 1])
        x_ob_prob = UpSampling2D(size=(4, 4))(x_ob_prob)
        x_ob_prob = tf.squeeze(x_ob_prob)
        return x_out_logi, x_ob_prob
        """

        return x_out_log

class BaselineNet():
    def __init__(self):
        return
    def baseline_reward(self, lstm_feature, keep_prob):
        # estimate Q value from lstm output state
        x = Dense(32)(lstm_feature)
        x = Activation('tanh')(x)
        x = Dropout(keep_prob)(x)
        reward = Dense(1)(x)
        return reward

def build_all(rgb_holder, depth_holder, rss_holder, phase_holder, lstm_nodes, n_unrolls, keep_prob, batch_size):
    # build conv head
    cc = ConvHead()
    x1, x2, x3, x4, x5_unpool, x5 = cc.build(rgb_holder, depth_holder)

    # encode feature with LSTM
    fe = FrameEncoder(lstm_nodes, n_unrolls, keep_prob)
    lstm_feature = fe.build(rss_holder, phase_holder, x5, batch_size)

    # decode feature with trans_conv
    fd = FrameDecoder()
    x_out_logi = fd.build(lstm_feature, x5_unpool, x4, x3, x2)
    #print(x_out_logi.get_shape().as_list())

    # baseline reward
    bn = BaselineNet()
    reward = bn.baseline_reward(lstm_feature, keep_prob)
    return x_out_logi, reward

