import numpy as np
import matplotlib.pyplot as plt


class Input():
    def __init__(self):
        self.rgb_dic = {}
        self.depth_dic = {}
        self.rss_dic = {}
        self.phase_dic = {}
        self.num_of_scen = 0
        return

    def add_from_npy(self, rgbd_src, rfid_src, id):
        rgbd_frames = np.load(rgbd_src)
        rfid_frames = np.load(rfid_src)
        print(np.shape(rgbd_frames))
        print(np.shape(rfid_frames))
        return
    def minmax_norm(self,array):
        max = np.max(array)
        min = np.min(array)
        norm = []
        for e in array:
            norm.append((e-min)/(max-min))
        return np.array(norm)
    def analysis(self,rgbd_src, rfid_src):
        rgbd_frames = np.load(rgbd_src)
        rfid_frames = np.load(rfid_src)
        depth = rgbd_frames[100:,0,:,:,:]
        color = rgbd_frames[100:,1,:,:,:]
        distance = rfid_frames[100:]

        print(np.shape(depth))
        print(np.shape(color))
        pixel_depth = []
        dis = []
        pixel_x = [263,266,273,279,290,304,310,320,331,343,347,355,358,360,360,358,351,342,318,306,297,294,261,248]
        pixel_y = [209,209,209,209,209,209,209,209,209,209,209,209,209,209,211,210,209,209,205,207,206,205,202,201]

        for i in range(len(pixel_x)):
            pixel_depth.append(depth[i][pixel_x[i]][pixel_y[i]][0])
            dis.append(distance[i])
        pixel_depth = np.array(pixel_depth)
        dis = np.array(dis)
        print(pixel_depth)
        print(dis)
        x = np.arange(len(dis))
        pixel_depth = self.minmax_norm(pixel_depth)
        dis = self.minmax_norm(dis)
        plt.figure(1)
        plt.plot(x,pixel_depth)
        plt.plot(x,dis)

        plt.figure(2)
        y = np.arange(len(distance))
        plt.plot(y, distance)


        plt.figure(3)
        imgplot = plt.imshow(color[23])
        plt.figure(4)
        imgplot = plt.imshow(depth[23])
        plt.show()

if __name__ == '__main__':
    input = Input()
    input.analysis('rgbd.npy', 'rfid.npy')