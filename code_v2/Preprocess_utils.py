import os
import math
import scipy.constants
import numpy as np
import matplotlib.pyplot as plt
import imageio
import csv
import CONFIG
import cv2

# finds whether you need to shift frequency up or down 2pi
def shift_2pi(phase_angle1, phase_angle2, threshold):
    if abs(phase_angle2 - phase_angle1) >= (1 - threshold) * 2 * math.pi and abs(
            phase_angle2 - phase_angle1) <= 2 * math.pi:
        return True
    else:
        return False

# finds whether you need to shift frequency up or down pi
def shift_pi(phase_angle1, phase_angle2, threshold):
    if abs(phase_angle2 - phase_angle1) >= (1 - threshold) * math.pi and abs(phase_angle2 - phase_angle1) <= (
            1 + threshold) * math.pi:
        return True
    else:
        return False


# finds total distance from rfid
def finddistance(frequency, phase_angle):
    return (phase_angle * scipy.constants.c) / (4 * math.pi * frequency)

# read all rgb images from folder
def readColorImage(subdir, resize = False):
    files = os.listdir(subdir)
    timestamp_color = []  # stores the equivalent rgb timestamps
    colormatrix = []  # stores all the rgb in this list
    for i, file in enumerate(sorted(files)):  # instantiates the colormatrix and timestamp color
        im = imageio.imread(subdir + file)
        if resize == True:
            im = cv2.resize(im, (CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT))
        colormatrix.append(im)
        timestamp_color.append((float(file[:-4])))  # reads the filename not including the .png
    colormatrix = np.array(colormatrix)
    return colormatrix

# read a single XYZ matrix
def readXYZmatrix(src):
    with open(src) as f:
        data = f.readlines()
    matrix = []
    for line in data:
        line = line[:-2]
        x, y, z = line.split()
        if x == '-∞':
            x = 0
        if y == '-∞':
            y = 0
        if z == '-∞':
            z = 0
        matrix.append([float(x), float(y), float(z)])
    matrix = np.array(matrix)
    matrix = np.reshape(matrix, (424, 512, 3))
    return matrix

# read depth matrix
def readDepthmatrix(src):
    with open(src) as f:
        data = f.readlines()
    matrix = []
    for line in data:
        matrix.append([int(line)])
    matrix = np.array(matrix)
    matrix = np.reshape(matrix, (424, 512))
    print(np.shape(matrix))
    print(np.max(matrix))
    print(np.min(matrix))
    print(matrix[200])
    plt.imshow(matrix, cmap='gray')
    plt.show()

# read all XYZ matrices
def readXYZImage(subdir, resize = False):
    files = os.listdir(subdir)
    timestamp_XYZ = []  # stores the equivalent depth timestamps
    XYZmatrix = []  # stores all the depth in this list
    for i, file in enumerate(sorted(files)):  # instantiates the depthmatrix and timestamp
        print(i)
        im = readXYZmatrix(subdir + file)
        if resize == True:
            im = cv2.resize(im, (CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT))
        XYZmatrix.append(im)
        timestamp_XYZ.append((float(file[:-4])))  # reads the filename not including the .txt
    XYZmatrix = np.array(XYZmatrix)
    timestamp_XYZ = np.array(timestamp_XYZ)
    return timestamp_XYZ, XYZmatrix

# read RFID file
def RFIDMatrix(src):
    ts_sys = []
    phase = []
    rss = []
    doppler = []
    ID = []
    ts = []
    # df = pd.read_csv(src)
    # print(df['Phase Angle(Radian)'])
    with open(src, "rt") as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(spamreader):
            current_phase = float(row['Phase Angle(Radian)'])
            phase.append(2 * math.pi - current_phase)
            ts_sys.append(float(row['TS']))
            ts.append(float(row['Timestamp']) / 1000)
            current_rss = float(row['RSS(dBm)'])
            rss.append(current_rss)
            current_doppler = float(row['Doppler Shift(Hz)'])
            doppler.append(current_doppler)
            current_ID = row['EPC']
            ID.append(current_ID)

    d1 = []
    for i in range(len(ts)):
        d1.append(ts[i] - ts[0])
    d1 = np.array(d1)
    timestamp_rfid = d1 + ts_sys[0] - CONFIG.SYNCHRONIZATION_BIAS
    timestamp_rfid = np.array(timestamp_rfid)
    return timestamp_rfid, phase, rss, doppler, ID

# separate rfid signal according to tag ID
def separate_signals(timestamp_rfid, phase, rss, doppler, ID):
    t_dic = {}
    phase_dic = {}
    rss_dic = {}
    dop_dic = {}
    for i in range(0, len(timestamp_rfid)):
        if t_dic.__contains__(ID[i]):
            t_dic[ID[i]].append(timestamp_rfid[i])
            phase_dic[ID[i]].append(phase[i])
            rss_dic[ID[i]].append(rss[i])
            dop_dic[ID[i]].append(doppler[i])
        else:
            t_dic[ID[i]] = []
            phase_dic[ID[i]] = []
            rss_dic[ID[i]] = []
            dop_dic[ID[i]] = []
            t_dic[ID[i]].append(timestamp_rfid[i])
            phase_dic[ID[i]].append(phase[i])
            rss_dic[ID[i]].append(rss[i])
            dop_dic[ID[i]].append(doppler[i])
    return t_dic, phase_dic, rss_dic, dop_dic

# Smooth the phase with 2pi and pi shift
def PhaseSmooth(phase, t1, t2, timestamp):
    shiftmul = 0
    SmoothPhase = []
    SmoothPhase.append(phase[0])
    for i in range(1, len(phase)):
        current_phase = phase[i] + shiftmul
        previous_phase = phase[i - 1] + shiftmul
        diff = abs(current_phase - previous_phase) / math.pi
        '''
        if diff < 0.3 or (diff < 1.3 and diff > 0.7) or diff > 1.5:
            pass
        else:
            print("alert", i, diff)
            print(timestamp[i]-timestamp[i-1])
        '''
        # print(timestamp[i] - timestamp[i - 1])
        shift = 0
        if shift_2pi(current_phase, previous_phase, t1):
            # print(i, "A")
            if current_phase - previous_phase < 0:
                # shift 2pi upwards
                shiftmul += 2 * math.pi
                shift = 2 * math.pi
            else:
                # shift downwards
                shiftmul -= 2 * math.pi
                shift = -2 * math.pi
        elif shift_pi(current_phase, previous_phase, t2):
            # print(i,"B")
            if current_phase - previous_phase < 0:
                # shift 2pi upwards
                shift = math.pi
                shiftmul += math.pi
            else:
                # shift downwards
                shift = -math.pi
                shiftmul -= math.pi
        current_phase += shift
        SmoothPhase.append(current_phase)
    return SmoothPhase

# calculate (relative) distance from phase
def PhasetoDist(phase):
    dist = []
    for ph in phase:
        d = (ph * scipy.constants.c) / (4 * math.pi * CONFIG.FREQUENCY)
        dist.append(d)
    return dist

# calculate the gradient of a signal
def Gradient(dis, timestamp):
    grad = []
    for i in range(len(dis) - 1):
        grad.append((dis[i + 1] - dis[i]) / (timestamp[i + 1] - timestamp[i]))
    return grad

#  read labels
def readHW(src):
    with open(src) as f:
        array = []
        for line in f:  # read rest of lines
            array.append([int(x) for x in line.split()])
    array = np.array(array)
    return array

# smooth the signal according to velocity and acceleration
def acceleration_smooth(dis, timestamp, th_a):
    velocity = Gradient(dis, timestamp)
    acceleration = Gradient(velocity, timestamp[1:])
    v_smooth = []
    dis_smooth = []
    v_smooth.append(velocity[0])
    for i in range(0, len(acceleration)):
        if math.fabs(acceleration[i]) > th_a:
            v_smooth.append(v_smooth[-1])
        else:
            v_smooth.append(velocity[i + 1])
    dis_smooth.append(dis[0])
    tem = dis[0]
    for i in range(len(v_smooth)):
        tem += v_smooth[i] * (timestamp[i + 1] - timestamp[i])
        dis_smooth.append(tem)
    dis_smooth = np.array(dis_smooth)
    return dis_smooth

# generate required camera data for learning
def camera_raw_data(color_subdir, XYZ_subdir, timestamp_bias, resize = False):
    color_matrix = readColorImage(color_subdir,resize)
    ts_XYZ, XYZ_matrix = readXYZImage(XYZ_subdir, resize)
    XYZ_matrix = XYZ_matrix + CONFIG.SENSOR_BIAS
    dismatrix = np.linalg.norm(XYZ_matrix, axis=3)
    camera_ts = ts_XYZ - timestamp_bias
    return color_matrix, dismatrix, camera_ts

# generate required rfid data (preprocessed) for learning
def RFID_raw_data(rfid_src, Tag_ID):
    timestamp_rfid, phase, rss, doppler, ID = RFIDMatrix(rfid_src)
    timestamp_bias = timestamp_rfid[0]
    timestamp_rfid = timestamp_rfid - timestamp_bias
    t_dic, phase_dic, rss_dic, dop_dic = separate_signals(timestamp_rfid, phase, rss, doppler, ID)

    #preprocess
    SmoothPhase = PhaseSmooth(phase_dic[Tag_ID], CONFIG.PI2_SHIFT_TH, CONFIG.PI_SHIFT_TH, t_dic[Tag_ID])
    dist = PhasetoDist(SmoothPhase)
    rfid_dist_smooth = acceleration_smooth(dist, t_dic[Tag_ID], CONFIG.ACCELERATION_RATE_TH * scipy.constants.g / 1e6)
    rfid_velocity = Gradient(rfid_dist_smooth, t_dic[Tag_ID])
    velocity_ts = t_dic[Tag_ID][1:]
    rfid_dist_ts = t_dic[Tag_ID]
    rss = rss_dic[Tag_ID]
    return rfid_dist_smooth, rfid_velocity, rfid_dist_ts, velocity_ts, timestamp_bias, rss


if __name__ == '__main__':

    """
    timestamp_XYZ, XYZmatrix = readXYZImage('./kinect/XYZmatrix/')
    np.save('XYZmatrix', XYZmatrix)
    np.save('dismatrix', dismatrix)
    np.save('timestamp_XYZ', timestamp_XYZ)

    """

    timestamp_rfid, phase, rss, doppler, ID = RFIDMatrix('./kinect/7.25.14.00.csv')

    timestamp_bias = timestamp_rfid[0]
    timestamp_rfid = timestamp_rfid - timestamp_bias
    t_dic, phase_dic, rss_dic, dop_dic = separate_signals(timestamp_rfid, phase, rss, doppler, ID)

    # time = np.arange(len(phase_dic[CONFIG.TAG_1]))
    # plt.figure('phase')
    # plt.plot(t_dic[CONFIG.TAG_1],phase_dic[CONFIG.TAG_1])
    # plt.figure('rss')
    # plt.plot(t_dic[CONFIG.TAG_1],rss_dic[CONFIG.TAG_1])

    timestamp_XYZ = np.load('timestamp_XYZ.npy')
    timestamp_XYZ = timestamp_XYZ - timestamp_bias
    dismatrix = np.load('dismatrix_unbias.npy')

    """
    plt.figure(1)
    x = np.arange(424)
    y = np.arange(512)
    x_1 = np.zeros(424)+235
    y_1 = np.zeros(512)+200
    for i in range(284, 200, -1):
        print(i)
        plt.imshow(dismatrix[i], cmap='gray')
        plt.plot(x_1,x)
        plt.plot(y, y_1)
        plt.show()

    #246 230 14 350-337
    #246 250 5 336-332
    #246 290 6 331-326
    #240 320 15 325-311
    #225 340 4 310-307
    #230 305 9 306-298
    #235 270 7 297-291
    #235 234 6 290-285
    #235 200 6 284-277
    """
    SmoothPhase = PhaseSmooth(phase_dic[CONFIG.TAG_1], 0.3, 0.3, t_dic[CONFIG.TAG_1])
    plt.figure('Sphase')
    plt.plot(t_dic[CONFIG.TAG_1], SmoothPhase)

    # XYZmatrix = np.load('XYZmatrix.npy')
    # XYZmatrix = XYZmatrix + CONFIG.SENSOR_BIAS
    # dismatrix = np.linalg.norm(XYZmatrix, axis=3)

    camera_dis_list = np.array([])

    xy_list = readHW('labeled_gnome_nose')

    dis_list_head = []
    for i in range(len(xy_list)):
        dis_list_head.append(dismatrix[i, xy_list[i][1], xy_list[i][0]])
    dis_list_head = np.array(dis_list_head)
    # print(dis_list_head)

    camera_dis_list = np.concatenate((camera_dis_list, dis_list_head))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[277:285, 200, 235]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[285:291, 234, 235]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[291:298, 270, 235]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[298:307, 305, 230]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[307:311, 340, 225]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[311:326, 320, 240]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[326:332, 290, 246]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[332:337, 250, 246]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[337:351, 230, 246]))
    camera_dis_list = np.concatenate((camera_dis_list, dismatrix[351:, 220, 253]))

    t1 = timestamp_XYZ[0: 179]
    t2 = timestamp_XYZ[277:]
    t_XYZ = np.concatenate((t1, t2))
    dist = PhasetoDist(SmoothPhase)
    dis_smooth = acceleration_smooth(dist, t_dic[CONFIG.TAG_1], 0.5 * scipy.constants.g / 1e6)
    plt.figure('distance')
    plt.plot(t_dic[CONFIG.TAG_1], dist)
    plt.plot(t_XYZ, camera_dis_list, 'orangered')
    plt.plot(t_dic[CONFIG.TAG_1], dis_smooth)

    rfid_dis_grad = Gradient(dist, t_dic[CONFIG.TAG_1])
    camera_dis_grad = Gradient(camera_dis_list, t_XYZ)

    rfid_dis_grad_smooth = Gradient(dis_smooth, t_dic[CONFIG.TAG_1])

    plt.figure('PhaseGrad')
    plt.plot(t_dic[CONFIG.TAG_1][1:], rfid_dis_grad)
    plt.plot(t_dic[CONFIG.TAG_1][1:], rfid_dis_grad_smooth)
    # plt.plot(t_XYZ[1:],camera_dis_grad,'orangered')
    zero = np.zeros(len(t_dic[CONFIG.TAG_1][:-1]))
    plt.plot(t_dic[CONFIG.TAG_1][1:], zero, 'black')

    a = Gradient(rfid_dis_grad, t_dic[CONFIG.TAG_1][1:])
    a_smooth = Gradient(rfid_dis_grad_smooth, t_dic[CONFIG.TAG_1][1:])
    plt.figure('a')
    plt.plot(t_dic[CONFIG.TAG_1][2:], a)
    plt.plot(t_dic[CONFIG.TAG_1][2:], a_smooth)

    plt.show()
