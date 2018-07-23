import CONFIG
from network import build_all
import tensorflow as tf
import numpy as np
from scipy import stats
# parameters
keep_prob = 0.4
learning_rate = 1e-3
epoch = 10
batch_trajectory = 2 # number of n_unroll trajectory, batchsize = batch_trajectory * n_unrolls
discount = 0.9

class Path:
    def __init__(self, reward_list):
        self.reward_list = reward_list
        return

    def get_re_array(self):
        return np.array(self.reward_list)

    def length(self):
        return len(self.reward_list)

class Memory:
    def __init__(self):
        self.replays = []
    def add_path(self, path):
        self.replays.append(path)

    def generate_re_batch(self):
        re_array_list = []
        for play in self.replays:
            re_array = play.get_re_array()
            re_array_list.append(re_array)
        return np.concatenate(tuple(re_array_list))

    def get_Q_value(self, discount):
        q_array = []
        for play in self.replays:
            re_array = play.get_re_array()
            re_array = reversed(re_array)
            q_list = []
            q = 0
            for reward in re_array:
                q = reward + q * discount
                q_list.append(q)
            q_list.reverse()
            q_array.extend(q_list)
        return  np.array(q_array)

    def get_adv_value(self, baseline_pred_value, discount):
        adv_array = []
        checkpoint = 0
        for play in self.replays:
            play_len = play.length()
            play_baseline = baseline_pred_value[checkpoint:(checkpoint+play_len)].copy()
            checkpoint += play_len
            re_play_baseline = reversed(play_baseline)
            re_array = play.get_re_array()
            re_array = reversed(re_array)
            adv_list = []
            value_next = 0
            adv = 0
            for reward, baseline_value in zip(re_array,re_play_baseline):
                #time-difference (TD) error
                TD_error = reward + value_next * discount - baseline_value
                adv = TD_error + discount * adv
                value_next = baseline_value
                adv_list.append(adv)
            adv_list.reverse()
            adv_array.extend(adv_list)
        #print(adv_array)
        return np.array(adv_array)

    def averaged_play_reward(self):
        reward_list = []
        for play in self.replays:
            re_array = play.get_re_array()
            reward_list.append(np.sum(re_array))
        reward_list = np.array(reward_list)
        return np.mean(reward_list)

def resize_frames():
    return

def reward_trajectory(action_frames, depth_frames, phase_frames):
    # calculate the gained rewards after taking each action in the trajectory
    reward = np.arange(len(action_frames))
    return reward

def policy_grad_training(rgb_frames, depth_frames, rss_frames, phase_frames):
    # rgb_frames, depth_frames, rss_frames, phase_frames are dictionaries
    # keys represent different scenarios, values are [n_frames, h, w, 3], [n_frames, h, w, 1]
    #[n_frames, rss], [n_frames, phase]

    # observation holders
    # Note: hidden state in LSTM is also an observation, which is defined in LSTM units
    rgb_holder = tf.placeholder(tf.float32, [None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 3])
    depth_holder = tf.placeholder(tf.float32, [None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1])
    rss_holder = tf.placeholder(tf.float32, [None, 1])
    phase_holder = tf.placeholder(tf.float32, [None, 1])
    batch_size = tf.placeholder(tf.int32, shape=())
    #action holder
    action_holder = tf.placeholder(shape=[None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH], name="ac", dtype=tf.int32)

    # learn action from state
    lstm_nodes = CONFIG.LSTM_LAYERS
    n_unrolls = CONFIG.N_UNROOLS
    # x_out_log: [batch, h, w, 2]
    x_out_log, baseline_value = build_all(rgb_holder, depth_holder, rss_holder, phase_holder, lstm_nodes, n_unrolls, keep_prob, batch_size)
    x_shape = x_out_log.get_shape().as_list()
    x_out_log_reshape = tf.reshape(x_out_log, tf.stack([batch_size*n_unrolls*x_shape[1]*x_shape[2], x_shape[3]]))
    # sampling sampled_ac:[batch, h, w]
    sampled_ac = tf.multinomial(x_out_log_reshape, 1)
    sampled_ac = tf.reshape(sampled_ac, tf.stack([batch_size*n_unrolls,x_shape[1],x_shape[2]]))

    # Likelihood of chosen action: L = -sum_i y_i * log(p_i), y is one-hot action vector, thus,log(p_i) = -L
    logprob_ac = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action_holder, logits=x_out_log)
    logprob_ac = tf.reduce_sum(logprob_ac, [1,2])

    # advantage place holder
    adv_holder = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

    # policy update
    # reward_n = sum_t log(p_t) * sum_t adv_t, d_theta reward_n = d_theta sum_t log(p_t) * sum_t adv_t
    # reward_n = -loss, gradient accent
    loss = -tf.reduce_mean(logprob_ac * adv_holder)
    policy_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Montecarlo state-dependent expected return baseline
    # the baseline method estimate the reward given the state
    baseline_value = tf.squeeze(baseline_value)

    baseline_target_value = tf.placeholder(shape=[None], name="target", dtype=tf.float32)
    baseline_loss = tf.nn.l2_loss(baseline_value - baseline_target_value)
    baseline_update = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)

    saver = tf.train.Saver()
    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epoch):
            rgb_batch = []
            depth_batch = []
            phase_batch = []
            rss_batch = []
            ac_batch = []
            memory = Memory()
            # randomly pick a scenario id array
            sid_arr = np.random.randint(len(rgb_frames), size=batch_trajectory)
            for sid in sid_arr:
                # sampling one n_unroll length episode

                # randomly choose a start
                start = np.random.randint(len(rgb_frames[sid])-n_unrolls + 1, size=1)[0]
                end = start + n_unrolls

                # observations in this episode
                rgb_unroll = rgb_frames[sid][start:end]
                depth_unroll = depth_frames[sid][start:end]
                phase_unroll = phase_frames[sid][start:end]
                rss_unroll = rss_frames[sid][start:end]

                # play the game
                ac_frames = sess.run(sampled_ac,
                                    feed_dict={rgb_holder: rgb_unroll, depth_holder: depth_unroll,
                                    rss_holder: rss_unroll, phase_holder: phase_unroll, batch_size : 1})
                rgb_batch.append(rgb_unroll)
                depth_batch.append(depth_unroll)
                phase_batch.append(phase_unroll)
                rss_batch.append(rss_unroll)
                ac_batch.append(ac_frames)

                # calculate the reward for each step
                reward_list = reward_trajectory(ac_frames, depth_frames, phase_frames)
                path = Path(reward_list)
                memory.add_path(path)

            # data batch for training
            batch_Q_array = memory.get_Q_value(discount)
            rgb_batch = np.concatenate(tuple(rgb_batch))
            depth_batch = np.concatenate(tuple(depth_batch))
            phase_batch = np.concatenate(tuple(phase_batch))
            rss_batch = np.concatenate(tuple(rss_batch))
            ac_batch = np.concatenate(tuple(ac_batch))

            # predict baseline Q value
            baseline_pred_value = sess.run(baseline_value, feed_dict={rgb_holder: rgb_batch, depth_holder: depth_batch,
                                    rss_holder: rss_batch, phase_holder: phase_batch, batch_size : batch_trajectory})

            # normlize predicted value to the real Q value scale
            baseline_pred_value = stats.zscore(baseline_pred_value) * np.std(batch_Q_array) + np.mean(batch_Q_array)

            # compute advantage
            batch_adv_array = memory.get_adv_value(baseline_pred_value, discount)
            batch_Q_array = baseline_pred_value + batch_adv_array

            # normalize
            norm_batch_adv_array = stats.zscore(batch_adv_array)
            norm_batch_Q_array = stats.zscore(batch_Q_array)

            # training baseline
            sess.run(baseline_update,
                     feed_dict={rgb_holder: rgb_batch, depth_holder: depth_batch,
                                rss_holder: rss_batch, phase_holder: phase_batch, batch_size : batch_trajectory,
                                baseline_target_value: norm_batch_Q_array})

            # training policy
            sess.run(policy_update,
                     feed_dict={rgb_holder: rgb_batch, depth_holder: depth_batch,
                                rss_holder: rss_batch, phase_holder: phase_batch, batch_size : batch_trajectory,
                                action_holder: ac_batch, adv_holder: norm_batch_adv_array})
            print(e)
            print(memory.averaged_play_reward())

if __name__ == '__main__':
    rgb_frames = np.zeros([128, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 3])
    depth_frames = np.zeros([128, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1])
    rss_frames = np.zeros([128, 1])
    phase_frames = np.zeros([128, 1])

    rgb_dic = {}
    depth_dic = {}
    rss_dic = {}
    phase_dic = {}

    rgb_dic[0] = rgb_frames
    rgb_dic[1] = rgb_frames
    rgb_dic[2] = rgb_frames

    depth_dic[0] = depth_frames
    depth_dic[1] = depth_frames
    depth_dic[2] = depth_frames

    rss_dic[0] = rss_frames
    rss_dic[1] = rss_frames
    rss_dic[2] = rss_frames

    phase_dic[0] = phase_frames
    phase_dic[1] = phase_frames
    phase_dic[2] = phase_frames

    policy_grad_training(rgb_dic, depth_dic, rss_dic, phase_dic)