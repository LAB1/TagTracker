import tensorflow as tf
from flownet import flownet, FLOW_SCALE
from ops import downsample
from losses import compute_losses, create_border_mask
import numpy as np
import matplotlib.pyplot as plt
from image_warp import image_warp
import os
import imageio
from scipy.misc import imread
import cv2


# REGISTER ALL POSSIBLE LOSS TERMS
LOSSES = ['occ', 'fb', 'ternary', 'smooth_2nd', 'photo']

def unsupervised_train(batch):
    normalization = [[104.920005, 110.1753, 114.785955]]
    channel_mean = tf.constant(normalization[0]) / 255.0
    im1, im2 = batch
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im_shape = tf.shape(im1)[1:3]
    im1_geo, im2_geo = im1, im2
    im1_photo, im2_photo = im1, im2

    loss_weights = {'ternary_weight' : 1.0, 'smooth_2nd_weight' : 3.0, 'fb_weight' : 0.2,
                    'occ_weight' : 12.4, 'photo_weight' : 1.0}

    border_mask = create_border_mask(im1, 0.1)

    # Images for loss comparisons with values in [0, 1] (scale to original using * 255)
    im1_norm = im1_geo
    im2_norm = im2_geo
    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    #build
    flows_fw, flows_bw = flownet(im1_photo, im2_photo, backward_flow=True,)


    flows_fw = flows_fw[-1]
    flows_bw = flows_bw[-1]


    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_patch_distances = [3, 2, 2, 1, 1]
    im1_s = downsample(im1_norm, 4)
    im2_s = downsample(im2_norm, 4)
    mask_s = downsample(border_mask, 4)
    final_flow_scale = FLOW_SCALE
    final_flow_fw = tf.image.resize_bilinear(flows_fw[0], im_shape) * final_flow_scale * 4
    final_flow_bw = tf.image.resize_bilinear(flows_bw[0], im_shape) * final_flow_scale * 4

    combined_losses = dict()
    combined_loss = 0.0
    for loss in LOSSES:
        combined_losses[loss] = 0.0

    flow_enum = enumerate(zip(flows_fw, flows_bw))

    for i, flow_pair in flow_enum:
        layer_name = "loss" + str(i + 2)

        flow_scale = final_flow_scale / (2 ** i)

        with tf.variable_scope(layer_name):
            layer_weight = layer_weights[i]
            flow_fw_s, flow_bw_s = flow_pair

            mask_occlusion = 'fb'
            assert mask_occlusion in ['fb', 'disocc', '']



            losses = compute_losses(im1_s, im2_s,
                                    flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                                    border_mask=mask_s,
                                    mask_occlusion=mask_occlusion,
                                    data_max_distance=layer_patch_distances[i])

            layer_loss = 0.0

            for loss in LOSSES:
                weight_name = loss + '_weight'
                layer_loss += loss_weights[weight_name] * losses[loss]
                combined_losses[loss] += layer_weight * losses[loss]

            combined_loss += layer_weight * layer_loss

            im1_s = downsample(im1_s, 2)
            im2_s = downsample(im2_s, 2)
            mask_s = downsample(mask_s, 2)

    regularization_loss = tf.losses.get_regularization_loss()
    final_loss = combined_loss + regularization_loss

    """
    warp_1 = image_warp(im1_photo, final_flow_bw)
    warp_1 = warp_1 + channel_mean

    warp_2 = image_warp(im2_photo, final_flow_fw)
    warp_2 = warp_2 + channel_mean

    dis_1, dis_2 = disbatch
    dis_1_warp = image_warp(dis_1, final_flow_bw)
    dis_2_warp = image_warp(dis_2, final_flow_fw)
    dis_diff_1 = dis_1_warp-dis_2
    dis_diff_2 = dis_2_warp - dis_1
    """
    return final_loss, final_flow_fw, final_flow_bw



if __name__ == '__main__':
    im1 = tf.placeholder(tf.float32, [None,None, None, 3])
    im2 = tf.placeholder(tf.float32, [None, None, None, 3])

    batch = (im1, im2)

    loss_, flow_fw_, flow_bw_ = unsupervised_train(batch)

    learning_rate = 1.0e-4
    op = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999,
                                 learning_rate=learning_rate)
    train_op = op.minimize(loss_)

    batch_size = 24

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/model.ckpt")
        max_e = 10000
        for e in range(0,max_e):
            block_index = np.arange(20)
            np.random.shuffle(block_index)
            for index in block_index:
                im_array = np.load('./train_data/DDDDDDDDDDDD0001/color_matrix.npy')
                im_array = np.pad(im_array, ((0, 0), (44, 44), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
                disarray = np.load('./train_data/DDDDDDDDDDDD0001/dismatrix.npy')
                disarray = np.expand_dims(disarray, -1)
                disarray = np.pad(disarray, ((0, 0), (44, 44), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
                batch_index = np.arange(len(im_array)-1)
                np.random.shuffle(batch_index)
                for i in range(0,len(batch_index)-batch_size, batch_size):
                    mini_batch_1 = batch_index[i:i+batch_size]
                    mini_batch_2 = mini_batch_1 + 1
                    """
                    plt.figure(1)
                    plt.imshow(im_array[mini_batch_1][8])
                    plt.figure(2)
                    plt.imshow(im_array[mini_batch_2][8])
                    plt.show()
                    """
                    feed_dict = {im1: im_array[mini_batch_1], im2: im_array[mini_batch_2],}

                    _, loss, flow_fw, flow_bw= sess.run([train_op, loss_, flow_fw_, flow_bw_],feed_dict=feed_dict,)

                    print(i,loss)

                    """
                    plt.figure(1)
                    plt.imshow(im_array[mini_batch_1][0])
                    plt.figure(2)
                    plt.imshow(im_array[mini_batch_2][0])
                    plt.figure(3)
                    plt.imshow(warp_1[0])
                    plt.figure(4)
                    plt.imshow(warp_2[0])
                    plt.figure(5)
                    plt.imshow(flow_fw[0,:,:,0])
                    plt.figure(6)
                    plt.imshow(flow_fw[0, :, :, 1])
                    plt.figure(7)
                    plt.imshow(dis_warp_1[0,:,:,0])
                    plt.figure(8)
                    plt.imshow(dis_warp_2[0,:,:,0])
                    plt.show()
                    break
                    """
                if e%1 == 0:
                    im1_array = []
                    im2_array = []
                    print(e,"----------------------save----------------------------")
                    save_path = saver.save(sess, "./model/model.ckpt")

