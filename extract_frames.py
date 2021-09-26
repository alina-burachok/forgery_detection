from distutils.command.config import config

import cv2
import os
import numpy
import subprocess
import math
from PIL import Image
#from PRNU import constants,prnu

def array2image(arr, mode='RGB'):
    """NumPy array to PIL Image"""
    arr = arr.swapaxes(1, 2).swapaxes(0, 2)
    arr[arr < 0] = 0
    arr[arr > 255] = 255
    arr = numpy.fix(arr).astype(numpy.uint8)
    return Image.fromarray(arr)

def extract_frames_from_video(path=None,type=None, frame=None):
    try:
        print ("Empieza : ")

        videoName = "/home/ali/Descargas/test1.mp4"
        cap = cv2.VideoCapture(videoName)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
        frames = []
        if cap.isOpened() and video_length > 0 :
            frame_ids = [0]
            if video_length >= 4 :
                frame_ids = [0,
                             round(video_length*0,25),
                             round(video_length*0,5),
                             round(video_length*0,75),
                             video_length - 1]
            count = 0
            success, image = cap.read()
            while success:
                if count in frame_ids :
                    frames.append(image)
                success, image = cap.read()

                cv2.imwrite("/home/ali/Documentos/img/imagen"+str(count)+".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 100))

                count += 1
        return frames

    except Exception as e:
        print('Error in function extract_frames_from_video:',e)
        raise

def get_frame_types(video):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video]).decode()
    frame_types = out.replace('pict_type=', '').split()
    print ('frame types : ', frame_types)
    return zip(range(len(frame_types)), frame_types)


def extract_i_frames(video):
    frame_types = get_frame_types(video)
    i_frames = [x[0] for x in frame_types if x[1] == 'I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = basename + '_i_frame_' + str(frame_no) + '.jpg'
            cv2.imwrite(outname, frame, (cv2.IMWRITE_JPEG_QUALITY, 100))
            print ('Saved: ' + outname)
        cap.release()
    else:
        print ('No I-frames in ' + video)

def num_frames_total(video):
    frame_types = get_frame_types(video)
    frames = [x[0] for x in frame_types if x[1] != '']
    return len(frames)


def extract_p_frames(video):
    frame_types = get_frame_types(video)
    p_frames = [x[0] for x in frame_types if x[1] == 'P']
    if p_frames:
        basename = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        for frame_no in p_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = basename + '_p_frame_' + str(frame_no) + '.jpg'
            cv2.imwrite(outname, frame, (cv2.IMWRITE_JPEG_QUALITY, 100))
            print ('Saved: ' + outname)
        cap.release()
    else:
        print ('No P-frames in ' + video)

def readfromFile(path):
    x = numpy.fromfile(path, dtype='uint8')
    #print (x)
    return x



def extract_SMB_SODB(videoName):
    command = '/home/ali/Descargas/FFmpeg-master/ffmpeg -y -debug mb_type -i ' + videoName + ' out.mp4 2> macro.txt'
    os.system(command)

def sumaTodo(m, totalFrames, E):

    suma = 0

    for j in range(math.floor(totalFrames/m)-1):

        for i in range(1, m):

            suma += E[j*m+i]

    return suma



if __name__ == '__main__':

    videoName = "/home/ali/Descargas/carC.avi"
    filepathSODB = "/tmp/sodb.txt"
    filepathSMB = "/tmp/smb.txt"
    filetotalMB = "/tmp/totalmb.txt"
    fileGOP = "/tmp/gop.txt"

    max1 = 0
    max2 = 0
    tmp = 0
    T = 5
    R = 0

    extract_SMB_SODB(videoName)

    listSODB = readfromFile(filepathSODB)
    listSMB = readfromFile(filepathSMB)
    totalMB = readfromFile(filetotalMB)
    gopSize = readfromFile(fileGOP)

    totalFrames = num_frames_total(videoName)
    print("frames total  " + str(totalFrames))
    print("frames totalMB  " + str(totalMB))
    print("frames gopSize  " + str(gopSize))

    cont = 0
    E = []

    for i in listSMB:

        media = totalMB[cont] / i
       # print("media no  " + str(media))
        mediaMB = listSODB[cont]
       # print(" media mb  " + str(mediaMB))

        print(" cont  " + str(cont))
        mod = numpy.mod(cont, gopSize[0])
        print(" mod  " + str(mod))

        if mod == 1:

            mediaMB = (listSODB[cont-1] + listSODB[cont+1]) / 2
            E.insert(cont, mediaMB / media)

           # print("e if 1  " + str(E))

           # print("media mb if 1 " + str(mediaMB))

        else:
            E.insert(cont, mediaMB * media)
            #print("e  if 2 " + str(E))
            #print("media mb if 2 " + str(mediaMB))

        cont = cont + 1

    fin = numpy.minimum(150, math.floor(totalFrames/10))

    print("fin : " + str(fin))

    for m in range(2, fin):

        suma = sumaTodo(m, totalFrames, E)

        res = (1/math.floor(totalFrames/m))*suma

        print("Res : " + str(res))

        if res > max1:

            tmp = max1
            max1 = res

        if tmp > max2:

            max2 = tmp

    print("max1 : " + str(max1) + "max2 : " + str(max2))

    print(str(max1-max2))

    if (max1-max2) > T:
        R = 1
        print("double compressed video")
    else:
        R = 0
        print("single compressed video")

    print("Terminado ")
