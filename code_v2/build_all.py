import tensorflow as tf
from flownet import flownet, FLOW_SCALE
from image_warp import image_warp
import numpy as np
import CONFIG
from Preprocessing import Input
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import Activation
from sklearn.preprocessing import normalize
import cv2

def UnFlowNet(batch):
    normalization = [[104.920005, 110.1753, 114.785955]]
    channel_mean = tf.constant(normalization[0]) / 255.0
    im1, im2 = batch
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im_shape = tf.shape(im1)[1:3]
    im1_photo, im2_photo = im1, im2

    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    # build
    flows_fw, flows_bw = flownet(im1_photo, im2_photo, backward_flow=True, )

    flows_fw = flows_fw[-1]
    flows_bw = flows_bw[-1]

    final_flow_scale = FLOW_SCALE
    final_flow_fw = tf.image.resize_bilinear(flows_fw[0], im_shape) * final_flow_scale * 4
    final_flow_bw = tf.image.resize_bilinear(flows_bw[0], im_shape) * final_flow_scale * 4

    warp_1 = image_warp(im1_photo, final_flow_bw)
    warp_1 = warp_1 + channel_mean

    return final_flow_fw, final_flow_bw, warp_1
    #return flows_fw[0], flows_bw[0]

def batch_warp(im_batch, dis_batch, n_unrolls, batchsize):
    # im_unroll [batch, unroll, H, W, C]
    im1_unroll, im2_unroll = im_batch
    dis1_unroll, dis2_unroll = dis_batch

    im_shape = im1_unroll.get_shape().as_list()
    im1 = tf.reshape(im1_unroll, tf.stack([batchsize*n_unrolls, im_shape[2], im_shape[3], im_shape[4]]))
    im2 = tf.reshape(im2_unroll, tf.stack([batchsize * n_unrolls, im_shape[2], im_shape[3], im_shape[4]]))
    dis1 = tf.reshape(dis1_unroll, tf.stack([batchsize * n_unrolls, im_shape[2], im_shape[3], 1]))
    #dis2 = tf.reshape(dis2_unroll, tf.stack([batchsize * n_unrolls, im_shape[2], im_shape[3], 1]))

    final_flow_fw, final_flow_bw = UnFlowNet([im1, im2])

    dis1_warp = image_warp(dis1, final_flow_bw)
    #dis2_warp = image_warp(dis2, final_flow_fw)

def attention_cell(im1_slice, im2_slice, dis1_slice, dis2_slice, pd_1, pd_2, ts_1, ts_2, p_pre):
    # im1_slice, im2_slice, dis1_slice, dis2_slice [b, H, W, C]
    # pd_1, pd_2 [b, 1] phase distance
    # ts_1, ts_2 [b, 1] timestamp
    # p_pre [b, H, W] probability map of previous frame
    # output: p_curr [b, H, W, C] probability map of current frame

    # step 1: calculate flow 1->2 flow_bw
    flow_fw, flow_bw = UnFlowNet([im1_slice, im2_slice])

    # step 2: warp distance map 1->2 and calculate velocity map 1->2
    dis_warp_12 = image_warp(dis1_slice, flow_bw)
    delta_dis = dis2_slice - dis_warp_12
    delta_t = tf.expand_dims(tf.expand_dims((ts_2-ts_1),-1),-1)
    v_map = delta_dis / delta_t

    # step 3: calculate phase distance velocity 1->2
    v_pd = (pd_2 - pd_1) / (ts_2 - ts_1)

    # step 4: calculate probability map q_map with attention model
    v_pd = tf.expand_dims(tf.expand_dims(v_pd,-1),-1)
    diff = v_pd-v_map

def attention_model(dis_map, pd, rbf_p):
    map_shape = tf.shape(dis_map)
    dis_map_unpack = tf.reshape(dis_map, tf.stack([map_shape[0] * map_shape[1], map_shape[2]]))
    #dis_map_norm = tf.nn.l2_normalize(dis_map_unpack, 1)
    dis_map_norm = dis_map_unpack

    #pd_norm = tf.nn.l2_normalize(pd)
    pd_norm = pd
    multiply = [map_shape[0] * map_shape[1]]
    pd_norm_tile = tf.tile(pd_norm, multiply)
    pd_norm = tf.reshape(pd_norm_tile, tf.stack([multiply[0], tf.shape(pd_norm)[0]]))

    # tragger
    #v_pd = tf.abs(pd[-1] - pd[0])
    #v_pd = Activation('relu')(v_pd - v_th)

    # calculate correlation
    #x = tf.reduce_sum(tf.multiply(dis_map_norm, pd_norm), 1)
    x = tf_corrcoef(dis_map_norm, pd_norm)
    x = Activation('relu')(x)
    #x = tf.log(x)

    v_dis = dis_map_norm[:,-1] - dis_map_norm[:,0]
    v_pd = pd_norm[:, -1] - pd_norm[:, 0]
    v_diff = v_pd - v_dis
    v_diff_rbf = tf.exp(-rbf_p * tf.square(v_diff))

    #x = tf.reduce_sum(tf.multiply(dis_map_norm-pd_norm, dis_map_norm-pd_norm), 1)

    """
    # softmax normalize
    exp_sum = tf.reduce_sum(tf.exp(x))
    x = tf.exp(x) / exp_sum
    """

    x = tf.reshape(x, tf.stack([map_shape[0], map_shape[1]]))
    v_diff_rbf = tf.reshape(v_diff_rbf, tf.stack([map_shape[0], map_shape[1]]))

    prob = v_diff_rbf*x + 0.1
    prob_log = tf.log(prob)
    return prob_log

def tf_corrcoef(t1, t2):
    t1_mean, t1_var = tf.nn.moments(t1, axes=-1, keep_dims=True)
    t2_mean, t2_var = tf.nn.moments(t2, axes=-1, keep_dims=True)
    cov_12 = tf.reduce_mean(tf.multiply((t1-t1_mean),(t2-t2_mean)),-1, keep_dims=True)
    corrcoef_12 = cov_12 / tf.multiply(tf.sqrt(t1_var), tf.sqrt(t2_var))
    return corrcoef_12


def online_warp(im1_array, im2_array, dis1_array, dis2_array, unroll_size, pd_array):
    im_holder1 = tf.placeholder(tf.float32, [None, None, None, 3])
    im_holder2 = tf.placeholder(tf.float32, [None, None, None, 3])
    dis_holder1 = tf.placeholder(tf.float32, [None, None, None, 1])
    dis_holder2 = tf.placeholder(tf.float32, [None, None, None, 1])
    dis1_warp_holder = tf.placeholder(tf.float32, [None, None, None, None])
    dis_fw_queue_holder = tf.placeholder(tf.float32, [None, None, None])
    pd_holder = tf.placeholder(tf.float32, [None])
    rbf_p = tf.constant([1000], dtype = tf.float32)
    h_map_holder = tf.placeholder(tf.float32, [None, None, None, 1])


    att_map_ = attention_model(dis_fw_queue_holder, pd_holder, rbf_p)

    batch = (im_holder1, im_holder2)

    flow_fw_, flow_bw_, im1_warp_ = UnFlowNet(batch)

    dis1_warp = image_warp(dis_holder1, flow_bw_)

    dis1_warp_queue = image_warp(dis1_warp_holder, flow_bw_)

    h_map_warp = image_warp(h_map_holder, flow_bw_)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        #sess.run(tf.global_variables_initializer())
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver.restore(sess, "./model/model.ckpt")

        count = 0
        dis_fw_list = []
        pd_start = 0
        pd_end = 1
        h_map_buff = np.zeros(np.shape(dis1_array[0]))
        for im1,im2,dis1,dis2 in zip(im1_array, im2_array, dis1_array, dis2_array):
            print(count)
            if count == 0:
                feed_dict = {im_holder1: [im1], im_holder2: [im2],
                             dis_holder1: [dis1], dis_holder2: [dis2], h_map_holder: [h_map_buff]}
                dis_fw, h_map = sess.run([dis1_warp,h_map_warp], feed_dict=feed_dict, )
                count += 1
                dis_fw_list = np.concatenate((dis_fw[0],dis2), -1)
                pd_end += 1

            else:
                feed_dict = {im_holder1: [im1], im_holder2: [im2],
                             dis_holder1: [dis1], dis_holder2: [dis2],
                             dis1_warp_holder: [dis_fw_list], h_map_holder: [h_map_buff]}
                dis_fw_queue, flow_bw, h_map, im1_warp = sess.run([dis1_warp_queue,flow_bw_,h_map_warp, im1_warp_], feed_dict=feed_dict, )
                count += 1
                dis_fw_list = np.concatenate((dis_fw_queue[0],dis2), -1)
                pd_end += 1

            if count > unroll_size:
                dis_fw_list = dis_fw_list[:,:,1:]
                pd_start += 1

        #if count % 20 == 0:
            # attention model
            # dis_fw_list [H, W, LQ]
            # pd_queue [LQ]
            pd_queue = pd_array[pd_start:pd_end]
            dis_fw_list_unbias = dis_fw_list - np.expand_dims(dis_fw_list[:,:,0], -1)
            pd_queue_unbias = pd_queue - pd_queue[0]
            att_feed_dict = {dis_fw_queue_holder: dis_fw_list_unbias, pd_holder: pd_queue_unbias}
            att_map = sess.run(att_map_, feed_dict=att_feed_dict, )
            weight = 0
            if np.abs(pd_queue[-1] - pd_queue[0]) > 0.01:
                weight = np.abs(pd_queue[-1] - pd_queue[0])

            """
            if count <= 50:
                h_map_buff = h_map[0] + np.expand_dims(att_map,-1) * weight
            else:
                h_map_buff = h_map[0]
            """
            h_map_buff = h_map[0] + np.expand_dims(att_map, -1) * weight

            #temp = np.squeeze(h_map_buff, -1)

            """
            if count > 214:
                fill_map = refine(np.squeeze(h_map_buff, -1), np.squeeze(dis2, -1), 512, 512, True)
            else:
                fill_map = refine(np.squeeze(h_map_buff, -1), np.squeeze(dis2, -1), 512, 512)
            """
            fill_map = refine(np.squeeze(h_map_buff, -1), np.squeeze(dis2, -1), 512, 512)
            fill_map = np.expand_dims(fill_map, -1)
            h_map_buff = fill_map


            #print(np.shape(h_map))
            #print(np.shape(h_map_buff))
            """
            plt.figure(1)
            plt.plot(pd_queue_unbias, 'black')
            plt.plot(dis_fw_list_unbias[291,224,:], 'red')
            plt.plot(dis_fw_list_unbias[268, 279, :],'green')
            plt.plot(dis_fw_list_unbias[94, 130, :],'blue')
            plt.plot(dis_fw_list_unbias[325, 231, :],'yellow')
            
            plt.figure(3)
            plt.plot(dis_fw_list[311,244,:], 'red')
            """
            if count % 10 == 0:
            #if count == 86:
                """
                plt.figure('im1')
                plt.imshow(im1)

                plt.figure('im2')
                plt.imshow(im2)

                plt.figure('im1_w')
                plt.imshow(im1_warp[0])

                plt.figure('h_map')
                plt.imshow(np.squeeze(h_map_buff,-1))

                plt.figure('att_map')
                plt.imshow(att_map)
                
                plt.figure(4)
                plt.imshow(flow_bw[0,:,:,0])
                plt.figure(5)
                plt.imshow(flow_bw[0, :, :, 1])
                
                plt.figure(1)
                plt.plot(pd_queue_unbias, 'black')
                plt.plot(dis_fw_list_unbias[291,224, :], 'red')
                #plt.plot(dis_fw_list_unbias[236, 374, :],'green')

                
                plt.figure(6)
                plt.imshow(dis1[:,:,0])
                plt.figure(7)
                #print(np.shape(dis_fw_list))
                plt.imshow(dis_fw_list[:,:,-1])
                plt.figure(8)
                plt.imshow(dis_fw_list[:,:,-2])
                """


                plt.figure('att_map')
                plt.imshow(att_map)
                plt.figure('dis2')
                plt.imshow(dis2[:, :, 0])
                plt.figure('h_map')
                plt.imshow(np.squeeze(h_map_buff, -1))
                plt.show()





def refine(h_map, dis_map, H, W, is_debug = False):
    h_map[h_map == np.inf] = np.min(h_map)
    p_map = np.exp(h_map)
    p_map = (p_map-np.min(p_map)+1e-6)/(np.max(p_map)-np.min(p_map)+1e-3)
    #p_map = (p_map - np.mean(p_map) + 1e-6) / (np.std(p_map) + 1e-3)
    if is_debug:
        plt.imshow(p_map)
        plt.show()
    p_map = p_map > 0.3
    p_map = p_map.astype(int)
    if np.sum(p_map) <= 0:
        return h_map
    rmin, rmax, cmin, cmax = bbox2(p_map)
    width = min(int(0.5 * (cmax - cmin)),20)
    height = min(int(0.5 * (rmax - rmin)),20)

    rmin = max(rmin-height, 0)
    rmax = min(rmax+height, H)
    cmin = max(cmin-width, 0)
    cmax = min(cmax+width, W)
    h_crop = crop(h_map, rmin, rmax, cmin, cmax)
    dis_crop = crop(dis_map, rmin, rmax, cmin, cmax)
    p_map_crop = crop(p_map, rmin, rmax, cmin, cmax)

    #histr = cv2.calcHist([dis_crop], [0], None, [100], [0, 10])
    hist, bins = np.histogram(dis_crop.ravel(), 450, [0.5, 5])
    hist = hist - np.sum(hist)/256
    ranges = hist_seg(hist)
    ranges = np.array(ranges)
    ranges = ranges * 0.01 + 0.5

    if is_debug:
        #print(np.sum(hist))
        print(ranges)
        plt.figure(1)
        plt.plot(hist)
        plt.show()

    layers = segment(dis_crop, ranges)
    smooth_map = seg_smooth(layers, h_crop, p_map_crop, is_debug)
    fill_map = fill(h_map, rmin, rmax, cmin, cmax, smooth_map)

    return fill_map

def fill(h_map, rmin, rmax, cmin, cmax, smooth_map):
    left = h_map[rmin : rmax, 0:cmin]
    right = h_map[rmin: rmax, cmax:]
    up = h_map[0 : rmin, :]
    down = h_map[rmax:, :]
    fill_map = np.concatenate((left, smooth_map, right), 1)
    fill_map = np.concatenate((up, fill_map, down), 0)
    return fill_map

def seg_smooth(layers, h_crop, p_map_crop, is_debug = False):
    layer_pad = np.full(np.shape(h_crop), 0)
    smoothed_layers = np.full(np.shape(h_crop), 0)
    for layer in layers:
        intersect = p_map_crop * layer
        sub = layer - intersect
        ratio = np.count_nonzero(intersect) / (np.count_nonzero(intersect)+np.count_nonzero(sub))
        if is_debug:
            plt.figure(1)
            plt.imshow(p_map_crop)
            plt.figure(2)
            plt.imshow(layer)
            plt.show()
            print(ratio)
        if ratio > 0.3:
            min_value = np.min(intersect * h_crop)
            full_min = np.full(np.shape(h_crop), min_value)
            full_min = full_min * sub
            layer_pad = layer_pad + intersect * h_crop + full_min
            smoothed_layers += layer
    if np.sum(smoothed_layers) == 0:
        return h_crop
    intersect = p_map_crop * smoothed_layers
    p_map_sub = p_map_crop - intersect
    sub_mean = np.min(h_crop)

    full_mean = np.full(np.shape(h_crop), sub_mean)
    full_max = p_map_sub * full_mean
    unpad = 1 - (smoothed_layers+p_map_sub)
    smooth_map = unpad * full_mean + layer_pad + full_max
    if is_debug:
        plt.figure('unpad')
        plt.imshow(unpad)
        plt.figure('unpad * h_crop')
        plt.imshow(unpad * h_crop)
        plt.figure('h_crop')
        plt.imshow(h_crop)
        plt.figure('p_map_crop')
        plt.imshow(p_map_crop)
        plt.show()
        print(sub_mean)
    return smooth_map

def segment(dis_crop, ranges):
    layers = []
    for r in ranges:
        layer = (dis_crop >= r[0]) & (dis_crop <= r[1])
        layer = layer.astype(int)
        layers.append(layer)
    return layers

def hist_seg(hist):
    ranges = []
    start = 0
    state = 0
    count = 0
    for i in range(len(hist)):
        if state == 0:
            if hist[i] <= 0:
                continue
            else:
                start = i
                state = 1
                count += hist[i]
        elif state == 1:
            if hist[i] > 0:
                count += hist[i]
            else:
                if count >= 64:
                    end = i
                    ranges.append((start,end))
                    state = 0
                    count = 0
                else:
                    state = 0
                    count = 0
    return ranges

def crop(map, rmin, rmax, cmin, cmax):
    return map[rmin:rmax, cmin:cmax]

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def interpld_phase(phase, rfid_ts, camera_ts):
    phase_squeeze = np.squeeze(phase, -1)
    interpld_phase = interp1d(rfid_ts, phase_squeeze, kind='cubic')

    phase_inter = interpld_phase(camera_ts)

    return phase_inter

if __name__ == '__main__':
    """
    im1_array = np.zeros((100, 512,512,3))
    im2_array = np.zeros((100, 512, 512, 3))
    dis1_array = np.zeros((100, 512, 512, 1))
    dis2_array = np.zeros((100, 512, 512, 1))
    pd_array = np.zeros((200))
    unroll_size = 10
    online_warp(im1_array, im2_array, dis1_array, dis2_array, unroll_size, pd_array)
    """

    input = Input()
    save_dir = './train_data/'
    Tag_ID = CONFIG.TAG_2
    resize = False
    rgb_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'color_matrix.npy'
    XYZ_dist_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'dismatrix.npy'
    phase_dist_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'rfid_dist_smooth.npy'
    rss_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'rss.npy'
    camera_ts_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'camera_ts.npy'
    rfid_ts_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'rfid_dist_ts.npy'
    rfid_velocity_ts_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'velocity_ts.npy'
    rfid_velocity_src = save_dir + Tag_ID + '_' + str(resize) + '_3/' + 'rfid_velocity.npy'
    input.add_from_npy(rgb_src, XYZ_dist_src, phase_dist_src, rss_src, camera_ts_src, rfid_ts_src,
                       rfid_velocity_ts_src, rfid_velocity_src, 0)

    rgb_frames, depth_frames, phase_frames, camera_ts, rfid_ts = input.att_training_inputs()


    phase_inter = interpld_phase(phase_frames[0], rfid_ts[0], camera_ts[0])

    im_array = np.pad(rgb_frames[0], ((0, 0), (44, 44), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
    dis_array = np.pad(depth_frames[0], ((0, 0), (44, 44), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))

    im1_array = im_array[50:-1]
    im2_array = im_array[51:]
    dis1_array = dis_array[50:-1]
    dis2_array = dis_array[51:]
    unroll_size = 5
    #plt.plot(camera_ts[0][0:], phase_inter[0:])
    #plt.show()
    online_warp(im1_array, im2_array, dis1_array, dis2_array, unroll_size, phase_inter[50:])




