import numpy as np
from Preprocess_utils import camera_raw_data
from Preprocess_utils import RFID_raw_data
from Preprocess_utils import readHW
from Preprocess_utils import Gradient
import CONFIG
import matplotlib.pyplot as plt

class Input():
    def __init__(self):
        self.rgb_dic = {}
        self.XYZ_dist_dic = {}
        self.rss_dic = {}
        self.phase_dist_dic = {}
        self.camera_ts = {}
        self.rfid_ts = {}
        self.rfid_velocity_ts = {}
        self.num_of_scen = 0
        self.rfid_velocity = {}
        return

    def add_from_npy(self, rgb_src, XYZ_dist_src, phase_dist_src, rss_src, camera_ts_src, rfid_ts_src,
                     rfid_velocity_ts_src, rfid_velocity_src, id):
        self.rgb_dic[id] = np.load(rgb_src)[:-10]
        self.XYZ_dist_dic[id] = np.expand_dims(np.load(XYZ_dist_src),3)[:-10]
        self.phase_dist_dic[id] = np.expand_dims(np.load(phase_dist_src),1)
        self.rss_dic[id] = np.expand_dims(np.load(rss_src),1)
        self.camera_ts[id] = np.load(camera_ts_src)[:-10]
        self.rfid_ts[id] = np.load(rfid_ts_src)
        self.rfid_velocity_ts[id] = np.load(rfid_velocity_ts_src)
        self.rfid_velocity[id] = np.load(rfid_velocity_src)
        self.num_of_scen += 1
        return

    def plots(self,id):
        camera_dis_list = np.array([])

        #xy_list = readHW('labeled_gnome_nose')
        '''
        dis_list_head = []
        for i in range(len(xy_list)):
            dis_list_head.append(self.XYZ_dist_dic[id][i, xy_list[i][1], xy_list[i][0]])
        dis_list_head = np.array(dis_list_head)
        # print(dis_list_head)

        camera_dis_list = np.concatenate((camera_dis_list, dis_list_head))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][277:285, 200, 235]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][285:291, 234, 235]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][291:298, 270, 235]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][298:307, 305, 230]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][307:311, 340, 225]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][311:326, 320, 240]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][326:332, 290, 246]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][332:337, 250, 246]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][337:351, 230, 246]))
        camera_dis_list = np.concatenate((camera_dis_list, self.XYZ_dist_dic[id][351:, 220, 253]))

        t1 = self.camera_ts[id][0: 179]
        t2 = self.camera_ts[id][277:]
        t_XYZ = np.concatenate((t1, t2))
        '''
        plt.figure('distance')
        plt.plot(self.rfid_ts[id], self.phase_dist_dic[id])
        #plt.plot(t_XYZ, camera_dis_list, 'orangered')

        plt.figure('velocity')
        plt.plot(self.rfid_velocity_ts[id], self.rfid_velocity[id])
        #camera_velocity = Gradient(camera_dis_list, t_XYZ)
        #plt.plot(t_XYZ[1:], camera_velocity, 'orangered')
        zero = np.zeros(len(self.rfid_velocity_ts[id]))
        plt.plot(self.rfid_velocity_ts[id], zero, 'black')
        plt.show()

    def training_inputs(self):
        sampled_phase_dist_dic = {}
        sampled_rss_dist_dic = {}

        for id in self.phase_dist_dic.keys():
            phase_dist_list = []
            rss_list = []
            for ts in self.camera_ts[id]:
                diff = np.abs(self.rfid_ts[id] - ts)
                index = np.argmin(diff)
                phase_dist_list.append(self.phase_dist_dic[id][index])
                rss_list.append(self.rss_dic[id][index])
            sampled_phase_dist_dic[id] = phase_dist_list
            sampled_rss_dist_dic[id] = rss_list

        return self.rgb_dic, self.XYZ_dist_dic, sampled_phase_dist_dic, sampled_rss_dist_dic, \
               self.rfid_velocity, self.camera_ts, self.rfid_velocity_ts

    def att_training_inputs(self):

        return self.rgb_dic, self.XYZ_dist_dic, self.phase_dist_dic, self.camera_ts, self.rfid_ts

def RawData_Save(color_subdir, XYZ_subdir, rfid_src, Tag_ID, save_dir, resize = False):
    #load RFID raw data
    rfid_dist_smooth, rfid_velocity, rfid_dist_ts, velocity_ts, timestamp_bias, rss = RFID_raw_data(rfid_src, Tag_ID)


    #load camera data
    color_matrix, dismatrix, camera_ts = camera_raw_data(color_subdir, XYZ_subdir, timestamp_bias, resize)

    np.save(save_dir + Tag_ID + '_' + str(resize) +'/' +'rfid_dist_smooth.npy',rfid_dist_smooth)
    np.save(save_dir + Tag_ID + '_' + str(resize) + '/' + 'rfid_velocity.npy',rfid_velocity)
    np.save(save_dir + Tag_ID + '_' + str(resize) + '/' + 'rfid_dist_ts.npy',rfid_dist_ts)
    np.save(save_dir + Tag_ID + '_' + str(resize) + '/' + 'velocity_ts.npy', velocity_ts)
    np.save(save_dir + Tag_ID + '_' + str(resize) + '/' + 'rss.npy', rss)
    np.save(save_dir + Tag_ID + '_' + str(resize) + '/' + 'color_matrix.npy', color_matrix)
    np.save(save_dir + Tag_ID + '_' + str(resize) + '/' + 'dismatrix.npy',dismatrix)
    np.save(save_dir + Tag_ID + '_' + str(resize) + '/' + 'camera_ts.npy',camera_ts)

if __name__ == '__main__':
    RawData_Save('./kinect_data/kinect_8.21.13.10/color/', './kinect_data/kinect_8.21.13.10/XYZmatrix/', './rfid_data/8.21.13.10.csv', CONFIG.TAG_2, './train_data/')
    """
    input = Input()
    save_dir = './train_data/'
    Tag_ID = CONFIG.TAG_2
    resize = False
    rgb_src = save_dir + Tag_ID +'_' + str(resize) +  '/' + 'color_matrix.npy'
    XYZ_dist_src = save_dir + Tag_ID +'_' + str(resize) + '/' + 'dismatrix.npy'
    phase_dist_src = save_dir + Tag_ID +'_' + str(resize) + '/' + 'rfid_dist_smooth.npy'
    rss_src = save_dir + Tag_ID  +'_' + str(resize) + '/' + 'rss.npy'
    camera_ts_src = save_dir + Tag_ID +'_' + str(resize) + '/' + 'camera_ts.npy'
    rfid_ts_src = save_dir + Tag_ID + '_' + str(resize) + '/' + 'rfid_dist_ts.npy'
    rfid_velocity_ts_src = save_dir + Tag_ID + '_' + str(resize) + '/' + 'velocity_ts.npy'
    rfid_velocity_src = save_dir + Tag_ID + '_' + str(resize) + '/' + 'rfid_velocity.npy'
    input.add_from_npy(rgb_src, XYZ_dist_src, phase_dist_src, rss_src, camera_ts_src, rfid_ts_src,
                       rfid_velocity_ts_src, rfid_velocity_src, 0)
    input.plots(0)
    """