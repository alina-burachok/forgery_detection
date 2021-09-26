from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import math
import openpiv.tools as tools
import openpiv.process as piv
from PyAstronomy import pyasl

import subprocess
import os

from ffmpeg import video
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential

import os
import re
import sys
import json
import subprocess
import itertools
import uuid

from parse_qp import parse_qp_output

import cv2
import ffmpeg_debug_qp_parser as ffmpeg
from numpy.dual import svd

Tdup = 0.96
Tshuf = 0.95


def correlation(arr1, arr2):

    cor = np.correlate(arr1, arr2)
    #print("CORR : ", cor)
    #print("arr1 : ", arr1)
    #print("arr2 : ", arr2)
    return cor


def computeVFIsequences(videoName):

    #por cada dos fotogramas llamar a la funcion de procesamiento y guardar lo que devuelve en una matriz

    print("Empieza : ")

    cap = cv2.VideoCapture(videoName)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    frames = []

    print(video_length)
    if cap.isOpened() and video_length > 0:
        count = 0
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()

            count += 1


    vfih = []
    vfiv = []
    cont = 0
    sumU = 0#np.array(np.zeros(shape=len(video)), dtype='i')
    sumV = 0#np.array(np.zeros(shape=len(video)), dtype='i')

    while cont+1 < len(frames):

        # Converting color image to grayscale image
        grayA = cv2.cvtColor(frames[cont], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(frames[cont+1], cv2.COLOR_BGR2GRAY)

      #  print("Se han recuperado las imagenes", frame_a, frame_b)

        u, v, sig2noise = piv.extended_search_area_piv(grayA.astype(np.int32), grayB.astype(np.int32),
                                                         window_size=16, overlap=8, dt=0.75, search_area_size=64,
                                                         sig2noise_method='peak2peak')

        #x, y = piv.get_coordinates(image_size=video[cont].shape, window_size=24, overlap=12)

        #print("u: ", u[cont, cont])
        #print("v: ", v[cont, cont])

        for i in range(0, len(u)):
            for j in range(0, len(v)):

                #print("i : ", i)
                #print("j : ", j)

                #print("u : ", u[i, j])
                #print("v : ", v[i, j])

                if math.isnan(u[i, j]) == False:

                    sumU = float(sumU + u[i, j])

                if math.isnan(v[i, j]) == False:
                    sumV = float(sumV + v[i, j])

                #print(sumU)
                #print(sumV)

        #sumU = sumU.astype('int')
        #sumV = sumV.astype('int')

        print(sumU)
        print(sumV)

        #np.append(vfih, sumU)
        #np.append(vfiv, sumV)
        vfih.append(sumU)
        vfiv.append(sumV)


        cont = cont + 1
        print("Contador : ", str(cont), " de ", video_length)

        #tools.display_vector_field('/home/ali/Descargas/exp1_001.txt', scale=100, width=0.0025)


        #u1, v1, sig2noise = piv.extended_search_area_piv(
        #frame_a.astype(np.int32), frame_b.astype(np.int32),
        #window_size=24, overlap=12, dt=0.02, search_area_size=64,
        #sig2noise_method='peak2peak' )
    #print("vfih : ", vfih)
    #print("vfiv : ", vfiv)

    fps = cap.get(cv2.cv2.CAP_PROP_FPS)

    #frames.get(cv2.cv2.CAP_PROP_FPS)

    #print("FPS : ", fps)

    #print("Sale de computeVFIsequences ")

    return vfih, vfiv, frames, fps


def computeSampledVFI(vfih, vfiv):

    svfih = []
    svfiv = []

    i = 1

    #print("SVFI entra ", len(vfih))
    #print("SVFI entra ", len(vfiv))

    while i < len(vfih)-1 and i < len(vfiv)-1:

        svfih.insert(i-1, np.maximum(np.maximum(vfih[i-1], vfih[i]), vfih[i+1]))
        svfiv.insert(i-1, np.maximum(np.maximum(vfiv[i - 1], vfiv[i]), vfiv[i + 1]))

        i = i+1
    #print("SVFIH calculated : ", svfih)
    #print("SVFIV calculated : ", svfiv)

    return svfih, svfiv


def computeRelativeFactorSequences(svfih, svfiv):
    rfh = []
    rfv = []

    i = 1
    rfh.insert(0, 0.0)
    rfv.insert(0, 0.0)

    #print("RF entra ", len(svfih))
    #print("RF entra ", len(svfiv))

    while i < len(svfih)-1 and i < len(svfiv)-1:

        if svfih[i - 1] == 0:

            rfh.insert(i, 0.0)
        else:
            rfh.insert(i, (((svfih[i - 1] + svfih[i+1]) / (svfih[i - 1] * svfih[i+1])) * svfih[i]))

        if svfiv[i - 1] == 0:

            rfv.insert(i, 0.0)
        else:

            rfv.insert(i, (((svfiv[i - 1] + svfiv[i+1]) / (svfiv[i - 1] * svfiv[i+1])) * svfiv[i]))

        i = i+1

    #print("RFIH calculated : ", rfh)
    #print("RFIV calculated : ", rfv)
    return rfh, rfv


def assignAbnormalPoints(rfh, rfv):

    print("tp entra ")

    t1 = pyasl.generalizedESD(rfh, 2)

    #print("t1 : ", t1)

    t2 = pyasl.generalizedESD(rfv, 2)

    #print("t2 : ", t2)

    tp = t1[0]

    tparr = t1[1]+t2[1]

    print("tp = ", tp)
    #print("tp Array = ", tparr)

    return tp, tparr, t1[1], t2[1]


def vpaDetected(tp1, tp2):

    result = False
    ##count_I -> número de I-MBs en el frame n
    ##count_S -> número de S-MBs en el frame n

    tprimero1 = min(tp1[0], tp1[1])
    tprimero2 = max(tp1[0], tp1[1])

    tsegundo1 = min(tp2[0], tp2[1])
    tsegundo2 = max(tp2[0], tp2[1])

    result1 = False
    result2 = False

    imb, smb = extract_MB()

    #print("IMB: ", imb)
    #print("SMB: ", smb)

    if count_i(tprimero2-1, imb, smb) < count_i(tprimero2, imb, smb) and count_i(tprimero2, imb, smb) > count_i(tprimero2+1, imb, smb) and count_s(tprimero2-1, imb, smb) > count_s(tprimero2, imb, smb) and count_s(tprimero2, imb, smb) < count_s(tprimero2+1, imb, smb):

        result1 = True

        print(result1)


    if count_i(tsegundo2-1, imb, smb) < count_i(tsegundo2, imb, smb) and count_i(tsegundo2, imb, smb) > count_i(tsegundo2+1, imb, smb) and count_s(tsegundo2-1, imb, smb) > count_s(tsegundo2, imb, smb) and count_s(tsegundo2, imb, smb) < count_s(tsegundo2+1, imb, smb):

        result2 = True

        print(result2)

    print("VPA detected ")

    return result1, result2



def extract_MB():

    frame = 0
    imb_frame = 0
    smb_frame = 0

    imb = []
    smb = []

    # find todos los "frame=" en el fichero
        # leer el número de frame
        # leer el número de SMB y IMB del frame
    # si el sig frame no tiene IMB y SMB usar los valores del anterior

    aString = '' ## contenido del fichero?

    f = open("./mb.txt")

    linea = f.readline()

    while linea != "":

        if "frame=" in linea:

            arr = linea.split()

           # print(arr)

            index = 0

            for i, elem in enumerate(arr):

                #print("Elem : ", elem)

                if "frame=" in elem:
                    index = i
                    #print("index i : ", index)

            #index = arr.index("frame=")

            frame = arr[index+1]

            pos = 0
            while pos < len(arr):

                arrAux = arr[pos].split(":")
                #print("array aux : ")
                #print(arrAux)
                if "I:" in arr[pos]:
                    indexAux = arrAux.index("I")
                    imb_frame = arrAux[indexAux+1]

                if "SKIP:" in arr[pos]:
                    indexAux = arrAux.index("SKIP")
                    smb_frame = arrAux[indexAux+1]

                #print(linea)

                pos = pos + 1

        imb.insert(int(frame), imb_frame)
        smb.insert(int(frame), smb_frame)

        linea = f.readline()

    #print(imb)
    #print(smb)
    f.close()
    return imb, smb


def count_i(tp, imb, smb):

    contador = tp

    while contador < imb.__len__():

        if int(imb[tp]) > 0:
            #print("count i : ", imb[tp])
            return int(imb[tp])

        else:
            contador = contador + 1

    return 0


def count_s(tp, imb, smb):

    contador = tp

    while contador < smb.__len__():

        if int(smb[tp]) > 0:
            #print("count s : ", smb[tp])
            return int(smb[tp])

        else:
            contador = contador + 1

    return 0

def doubleDuplicationTest(tp, video, vfih, vfiv):

    corh = []
    corv = []
    frame_no = []

    #print("Entra en el doubleDuplicationTest")

    t1 = min(tp[0], tp[1])
    t2 = max(tp[0], tp[1])

    #print("t1 : ", t1)
    #print("t2 : ", t2)

    l = t2 - t1
    flagDup = 0
    t_VFIh = vfih[t1:t2]
    t_VFIv = vfiv[t1:t2]

    ic = 0
    i = 1
    p = 0

    #print("L : ", l)


    while i+l < len(video):

        if i < (t1-l+1) or i > t2:

            s_VFIh = vfih[i:i+l]
            s_VFIv = vfiv[i:i + l]

            #print("len corh : ", len(corh))
            #print("len corv : ", len(corv))

            #print("corh : ", corh)
            #print("corv : ", corv)

            #print("ic : ", ic)

            corh.insert(ic, correlation(s_VFIh, t_VFIh))
            corv.insert(ic, correlation(s_VFIv, t_VFIv))

            frame_no.insert(ic, i)

            ic = ic + 1

        i = i + 1
    if len(corh)>0:
        mcorh = max(corh)
    else:
        mcorh = 0.0
    if len(corv) > 0:
        mcorv = max(corv)
    else:
        mcorv = 0.0

    if mcorh > Tdup or mcorv > Tdup :

        flagDup = 1

        if mcorh > Tdup :
            p = mcorh
        if mcorv > Tdup :
            p = mcorv

        # l frames from frame_no[p] are the original frames used for duplication

    #print("Sale del doubleDuplicationTest", int(p), flagDup, l)

    return p, frame_no, flagDup, l #frame_no[p]


def doubleShuffleTest( tp, video, vfih, vfiv):

    corh = []
    corv = []
    frame_no = []
    Chvfi = []
    Cvvfi = []
    Shvfi = []
    Svvfi = []

    #print("Entra en el doubleShuffleTest")

    t1 = min(tp[0], tp[1])
    t2 = max(tp[0], tp[1])

    #print("t1 : ", t1)
    #print("t2 : ", t2)

    flagShuf = 0
    l = t2 - t1
    p = 0
    jc = 0

    #print("L : ", l)

    for j in range(1, l):
        for k in range(1, l):

            if j != k:

                print(" J y K : ", j, k)
                print(" JC : ", jc)
                Chvfi.insert(jc, vfih[j:k])
                Cvvfi.insert(jc, vfiv[j:k])

                jc = jc + 1

    SChvfi = np.sort(Chvfi)
    SCvvfi = np.sort(Cvvfi)

    ic = 0
    i = 1

    while i + l < len(video):

        if i < (t1 - l + 1) or i > t2 :

            jc = 0

            for j in range(1, l) :
                for k in range(1, l) :

                    if j != k :

                        Shvfi.insert(jc, vfih[j:k])
                        Svvfi.insert(jc, vfiv[j:k])

                        jc = jc + 1

            SShvfi = np.sort(Shvfi)
            SSvvfi = np.sort(Svvfi)

            corh.insert(ic, correlation(SShvfi, SChvfi))
            corv.insert(ic, correlation(SSvvfi, SCvvfi))

            frame_no.insert(ic, i)

            ic = ic + 1

        i = i + 1

    #print("CORH :", corh)
    #print("CORV :", corv)

    if len(corh) > 0:
        mcorh = max(corh)
    else:
        mcorh = 0.0
    if len(corv) > 0:
        mcorv = max(corv)
    else:
        mcorv = 0.0

    #print("MCORH :", mcorh)
    #print("MCORV :", mcorv)

    if mcorv > Tshuf or mcorh > Tshuf :

        flagShuf = 1

        if mcorh > Tshuf:
            p = mcorh
        if mcorv > Tshuf:
            p = mcorv

        # l frames from frame_no[p] are the original frames used for shuffling forgery

    #print("Sale del doubleShuffleTest", int(p), flagShuf, l)

    return p, frame_no, flagShuf, l

def frames ( posInicio, longitud, video) :

    elems = []
    cont = 0

    # desde video[posInicio] hasta llegar a longitud elementos
    for frame in video:
        elems.insert(elems, cont, video[posInicio])
        cont = cont + 1


    return elems


def singleDuplicationTest(tp, fps, video, vfih, vfiv):

    l = fps
    flagDup = 0
    ic1 = 0
    ic2 = 0
    i = 1
    cor1h = []
    cor1v = []
    cor2h = []
    cor2v = []
    frame1_no = []
    frame2_no = []
    t1 = 0
    t2 = 0
    C1 = 0

    ic1 = 0
    ic2 = 0
    #print("Entra en el singleDuplicationTest")

    #print("tp : ", tp)
    #print("L : ", l)

    C1hvfi = vfih[tp - l:tp]
    C1vvfi = vfiv[tp - l:tp]
    C2hvfi = vfih[tp:tp + l]
    C2vvfi = vfiv[tp:tp + l]

    while i + l < len(video):

        if i < (tp -l + 1) or i > tp :

            Shvfi = vfih[i:i+l]
            Svvfi = vfiv[i:i+l]

            cor1h.insert(ic1, correlation(Shvfi, C1hvfi))
            cor1v.insert(ic1, correlation(Svvfi, C1vvfi))

            frame1_no.insert(ic1, i)

            ic1 = ic1 + 1

        if i < tp or i > (tp + l) :

            Shvfi = vfih[i:i + l]
            Svvfi = vfiv[i:i + l]

            cor2h.insert(ic2, correlation(Shvfi, C2hvfi))
            cor2v.insert(ic2, correlation(Svvfi, C2vvfi))

            frame2_no.insert(ic2, i)

            ic2 = ic2 + 1

        i = i + 1

    mcor1h = max(cor1h)
    mcor1v = max(cor1v)
    mcor2h = max(cor2h)
    mcor2v = max(cor2v)

    mcorh = max(mcor1h, mcor2h)
    mcorv = max(mcor1v, mcor2v)

    flag = 1
    i = 1

    if mcorh == mcor1h and mcorv == mcor1v :

        if mcorh == mcor1h :
            p =mcor1h
        if mcorv == mcor1v :
            p = mcor1v

        cond = True

        while cond :

            if mcorh > Tdup or mcorv > Tdup :

                flagDup = 1

                if flag :

                    t1 = frame1_no[p]
                    t2 = frame1_no[p] + l
                    flag = 0

                else :

                    t1 = t2 - ( i * l)

                C1 = frames (video, [tp - (i*l)], l)
                i = i + 1

                Shvfi = vfih[t2-(i*l):(t2-(i*l))+l]
                Svvfi = vfiv[t2-(i*l):(t2-(i*l))+l]

                mcorh = correlation(Shvfi, C1hvfi)
                mcorv = correlation(Svvfi, C1vvfi)

            else :

                t1 = partitionClipC1(C1)

                cond = False

    else :

        p = mcor2h
        p = mcor2v

        cond2 = True

        while cond2 :

            if mcorh > Tdup or mcorv > Tdup :

                flagDup = 1

                if flag :

                    t1 = frame2_no[p]
                    t2 = frame2_no[p] + l
                    flag = 0

                else:

                    t2 = t1 + (i*l)

                C2 = frames(video, tp+(i*l), l)

                Shvfi = vfih[t1+(i*l):(t1+(i*l))+l]
                Svvfi = vfiv[t1+(i*l):(t1+(i*l))+l]

                i = i + 1

                mcorh = correlation(Shvfi, C2hvfi)
                mcorv = correlation(Svvfi, C2vvfi)

                # go to line 305

            else :

                t2 = partitionClipC2(C2)
                cond2 = False

    if flagDup :

        #frames from t1 to t2 are the original frames used for duplication

        #print("Sale del singleDuplicationTest", flagDup)

        return t1, t2, flagDup


def singleShuffleTest(tp, video):

    frames = []
    flagShuf = 0
    l = 0

    #print("Entra en el singleShuffleTest")

    #print("L : ", l)
    #print("tp : ", tp)

    ## if tp.length == 1 añadir otro valor a tp( punto de inicio)
    if len(tp) == 1:

        tp[1] = 0

    p, frames, flagShuf, l = doubleShuffleTest(tp, video)

    #print("Sale del sigleShuffleTest", p, flagShuf, l)

    return p, frames, flagShuf, l


def partition(C):

    Cl = 0
    Cr = 0

    arraySplit = np.array_split(C, 2)

    Cl = arraySplit[0]
    Cr = arraySplit[1]

    return Cl, Cr


def partitionClipC1(C, S, l, t1):

    Cl, Cr = partition(C)

    Sl, Sr = partition(S)

    mcorh, mcorv = correlation(Cr, Sr)

    flag = 1

    cond3 = True

    while cond3 :

        if mcorh > Tdup or mcorv > Tdup:

            l = l/2

            if flag :

                t1 = t1 -l

            Cll, Clr = partition(Cl)
            Sll, Slr = partition(Sl)

            if Cl.length() > 5 :

                mcorh, mcorv = correlation(Clr, Slr)

                if mcorh > Tdup or mcorv > Tdup:

                    Cl = Cll
                    Sl = Sll

                    flag = 1

                else :

                    Cl = Clr
                    Sl = Slr

                    flag = 0

            cond3 = False

        else :

            Cl = Cr
            Sl = Sr

            flag = 0

    return t1


def partitionClipC2(C, S, l, t2):

    Cl, Cr = partition(C)

    Sl, Sr = partition(S)

    mcorh, mcorv = correlation(Cl, Sl)

    flag = 1

    if mcorh > Tdup or mcorv > Tdup:

        l = l / 2

        if flag:
            t2 = t2 + l

        Crl, Crr = partition(Cr)
        Srl, Srr = partition(Sr)

        if Cr.length() > 5:

            mcorh, mcorv = correlation(Crl, Srl)

            if mcorh > Tdup or mcorv > Tdup:

                Cr = Crr
                Sr = Srr

                flag = 1

            else:

                Cr = Crl
                Sr = Srl

                flag = 0

    else:

        Cr = Cl
        Sr = Sl

        flag = 0

    return t2


def startAlgorithm(videoName):

    vfih, vfiv, frames, fps = computeVFIsequences(videoName)
    svfih, svfiv = computeSampledVFI(vfih, vfiv)
    rfh, rfv = computeRelativeFactorSequences(svfih, svfiv)
    tp, tparr, tp1, tp2 = assignAbnormalPoints(rfh, rfv)

    if tp == 0:

        result = "Vídeo original tp=0"

    else:

        if tp == 2:

            result1, result2 = vpaDetected(tp1, tp2)

            if not result1 and not result2:

                result = "Vídeo original vpa no detected"

            else:

                p, frames_new, flagDup, l = doubleDuplicationTest(tparr, frames, vfih, vfiv)

                if flagDup == 1:

                    result = "Manipulación detectada " + str(len(frames_new)) + " - " + str(flagDup) + " - " + str(l)

                else:

                    p, frames_new, flagShuf, l = doubleShuffleTest(tparr, frames, vfih, vfiv)

                    if flagShuf == 1:

                        result = "Manipulación detectada " + str(len(frames_new)) + " - " + str(flagShuf) + " - " + str(l)

                    else:

                        result = "Manipulación detectada "

        else:

            clipIni, clipFin, flagDup = singleDuplicationTest(tparr, fps, frames, vfih, vfiv)

            if flagDup == 1:

                result = "Manipulación detectada " + str(clipIni) + " - " + str(clipFin) + " - " + str(flagDup)

            else:

                clipIni, clipFin, flagShuf = singleShuffleTest(tparr, frames, vfih, vfiv)

                if flagShuf == 1:

                    result = "Manipulación detectada " + str(clipIni) + " - " + str(clipFin) +" - " + str(flagShuf)

                else:

                    result = "Manipulación detectada "

    return result

def extract_frames_from_video_by_position(path, pos, type=None, frame=None):

    cap = cv2.VideoCapture(path)

    cap.set(cv2.CAP_PROP_POS_FRAMES,pos)

    return cap.read()


def extract_frames_from_video(path, type=None, frame=None):
    try:
        print("Empieza : ")

        cap = cv2.VideoCapture(path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
        frames = []

        print(video_length)
        if cap.isOpened() and video_length > 0:
            frame_ids = [0]
            if video_length >= 4:
                frame_ids = [0,
                             round(video_length*0,25),
                             round(video_length*0,5),
                             round(video_length*0,75),
                             video_length - 1]
            count = 0
            success, image = cap.read()
            while success:
                #if count in frame_ids:
                frames.append(image)
                success, image = cap.read()

               # cv2.imwrite("/home/ali/Documentos/img/imagen"+str(count)+".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 100))

                count += 1
        return frames

    except Exception as e:
        print('Error in function extract_frames_from_video:', e)
        raise

def get_frame_types(video):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video]).decode()
    frame_types = out.replace('pict_type=', '').split()
    print ('frame types : ', frame_types)
    return zip(range(len(frame_types)), frame_types)


def extract_i_frames(video):
    frame_types = get_frame_types(video)

    frames = []

    i_frames = [x[0] for x in frame_types if x[1] == 'I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            frames.append(frame)
        cap.release()
    else:
        print ('No I-frames in ' + video)

    return frames

def extract_i_frames_positions(video):
    frame_types = get_frame_types(video)

    list = []
    cont = 0

    for x in frame_types:

        if x[1] == 'I':
            list.append(cont)

        cont = cont + 1

    return list

def extract_p_frames(video):
    frame_types = get_frame_types(video)

    frames = []

    i_frames = [x[0] for x in frame_types if x[1] == 'I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            frames.append(frame)
        cap.release()
    else:
        print ('No I-frames in ' + video)

    return frames


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def preprocess_data(frames):

    # imagenes a escala de grises
    # gaussian kernel low pass filter sobre cada imagen
    # gray-frame menos low pass filter

    filtered_frames = []

    for i in frames:

        gray = rgb2gray(i)

        gaus = cv2.GaussianBlur(gray, (3, 3), 0,)

        high = np.abs(np.subtract(gray, gaus))

        filtered_frames.append(high)

    return filtered_frames


def load_data():

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    videoOri = "./VideosOriginales/navidad.mp4"
    videoComp = "./VideosComprimidos/navidad.mp4"

    videoOri2 = "./VideosOriginales/bolas.mp4"
    videoComp2 = "./VideosComprimidos/bolas.mp4"

    videoOri3 = "./VideosOriginales/gato.mp4"
    videoComp3 = "./VideosComprimidos/gato.mp4"

    videoOri4 = "./VideosOriginales/nieve.mp4"
    videoComp4 = "./VideosComprimidos/nieve.mp4"

    frames = extract_i_frames_positions(videoOri)
    print("positions : ", frames)
    frames = extract_i_frames_positions(videoComp)
    print("positions : ", frames)

    # cargar video con 1 compresion
    # leer los I-frames
    # cargar video con doble compresion
    # leer los frames que están en la posicion de cada I-frame

    #framesOri = extract_frames_from_video(videoOri)
    framesComp = extract_frames_from_video(videoComp)

    iFrames = extract_i_frames_positions(videoOri)

    #I-relocated frames (double compression)
    for cont in iFrames:

        #gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]
        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_train.append(gray)
        y_train.append('1')

    # P frames (double compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]
        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_train.append(gray)
        y_train.append('0')

    # P frames (simple compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_train.append(gray)
        y_train.append('0')


    #framesOri = extract_frames_from_video(videoOri2)
    framesComp = extract_frames_from_video(videoComp2)

    iFrames = extract_i_frames_positions(videoOri2)

    # I-relocated frames (double compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_test.append(gray)
        y_test.append('1')

    # P frames (double compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_test.append(gray)
        y_test.append('0')

    # P frames (simple compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_test.append(gray)
        y_test.append('0')



    #framesOri = extract_frames_from_video(videoOri3)
    framesComp = extract_frames_from_video(videoComp3)

    iFrames = extract_i_frames_positions(videoOri3)

    # I-relocated frames (double compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_train.append(gray)
        y_train.append('0')

    # P frames (double compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_train.append(gray)
        y_train.append('0')

    # P frames (simple compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        #h, w = gray.shape
        #gray = np.resize(gray, (-1, h, w))
        x_train.append(gray)
        y_train.append('0')

    #framesOri = extract_frames_from_video(videoOri4)
    framesComp = extract_frames_from_video(videoComp4)

    iFrames = extract_i_frames_positions(videoOri4)

    # I-relocated frames (double compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        # h, w = gray.shape
        # gray = np.resize(gray, (-1, h, w))
        x_test.append(gray)
        y_test.append('1')

    # P frames (double compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        # h, w = gray.shape
        # gray = np.resize(gray, (-1, h, w))
        x_test.append(gray)
        y_test.append('0')

    # P frames (simple compression)
    for cont in iFrames:
        # gray = cv2.cvtColor(framesComp[cont], cv2.COLOR_BGR2GRAY)
        gray = framesComp[cont]

        # h, w = gray.shape
        # gray = np.resize(gray, (-1, h, w))
        x_test.append(gray)
        y_test.append('0')

    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

def cnn(videoName):

    mnist = tf.keras.datasets.mnist

    # cargar
    (x_train, y_train), (x_test, y_test) = load_data()
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape[0])
    #print(x_train.size())
    print(x_test.shape[0])
    #print(x_test.size())

    h, w, x = x_train[0].shape
    h2, w2, x = x_test[0].shape

    x_train = x_train.reshape(x_train.shape[0], h, w, 3)
    x_test = x_test.reshape(x_test.shape[0], h2, w2, 3)
    print(x_train)
    print(x_test)

    input_shape = (h, w, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    model = Sequential()

    ############################################################################################
    model.add(Conv2D(3, kernel_size=(5, 5), input_shape=input_shape, padding='valid'))

    model.add(Dropout(0.2))

    model.add(Dense(16, activation=tf.nn.relu))
    #model.add(Activation(tf.nn.relu))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid'))

    ############################################################################################

    model.add(Conv2D(16, kernel_size=(5, 5), input_shape=input_shape, padding='same'))

    model.add(Dropout(0.2))

    model.add(Dense(32, activation=tf.nn.relu))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same'))

    ############################################################################################

    model.add(Conv2D(32, kernel_size=(1, 1), input_shape=input_shape, padding='valid'))

    model.add(Dropout(0.2))

    model.add(Dense(64, activation=tf.nn.relu))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same'))

    ############################################################################################

    model.add(Conv2D(64, kernel_size=(1, 1), input_shape=input_shape, padding='same'))

    model.add(Dropout(0.2))

    model.add(Dense(64, activation=tf.nn.relu))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same'))

    ############################################################################################

    model.add(Conv2D(128, kernel_size=(1, 1), input_shape=input_shape, padding='same'))

    model.add(Dropout(0.2))

    model.add(Dense(128, activation=tf.nn.relu))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same'))

    ############################################################################################

    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers

    #model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation=tf.nn.relu))  # fully-connected???

    model.add(Dense(2, activation=tf.nn.relu))  # fully-connected???

    model.add(Dense(2, activation=tf.nn.softmax))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=1, batch_size=5) #, validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test)
    print("SCORES : ", scores)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    x_video = extract_frames_from_video(videoName)
    x_video = np.asarray(x_video)
    x_video = x_video.reshape(x_video.shape[0], h, w, 3)
    x_video = x_video.astype('float32')
    x_video /= 255

    scoresVideo = model.predict(x_video)

    print("SCORES VIDEO: ", scoresVideo)
    print("Accuracy: %.2f%%" % (scoresVideo[1] * 100))

    return (scoresVideo[1] * 100)


menu_options = {
    1: 'Ejecutar red neuronal con vídeos originales',
    2: 'Ejecutar red neuronal con vídeos comprimidos',
    3: 'Ejecutar algoritmo principal para Inserción al Inicio',
    4: 'Ejecutar algoritmo principal para Inserción al Medio',
    5: 'Ejecutar algoritmo principal para Inserción al Final',
    6: 'Ejecutar algoritmo principal para Duplicado al Inicio',
    7: 'Ejecutar algoritmo principal para Duplicado al Medio',
    8: 'Ejecutar algoritmo principal para Duplicado al Final',
    9: 'Ejecutar algoritmo principal para Mezclado al Inicio',
    10: 'Ejecutar algoritmo principal para Mezclado al Medio',
    11: 'Ejecutar algoritmo principal para Mezclado al Final',
    12: 'Exit'
}

submenu_options1 = {
    1: 'akiyo',
    2: 'bolas',
    3: 'core',
    4: 'gato',
    5: 'gatos',
    6: 'krav',
    7: 'navidad',
    8: 'nieve',
    9: 'Exit'
}

submenu_options2 = {
    1: 'bolas',
    2: 'bus',
    3: 'claire',
    4: 'core',
    5: 'gato',
    6: 'gatos',
    7: 'krav',
    8: 'navidad',
    9: 'nieve',
    10: 'Exit',
}

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

def print_submenu1():
    for key in submenu_options1.keys():
        print(key, '--', submenu_options1[key])

def print_submenu2():
    for key in submenu_options2.keys():
        print(key, '--', submenu_options2[key])


def ejecutar_cnn(videoName):

    video = extract_frames_from_video(videoName)

    data = preprocess_data(video)

    dobleCompresion = cnn(videoName)

    print("dobleCompresion : ", dobleCompresion)



def ejecutar_algoritmo(videoname):

    command = 'ffmpeg -y -debug mb_type -i ' + videoName + ' -strict -2 out.mp4 2> mb.txt'
    os.system(command)

    frames = extract_i_frames_positions(videoName)

    video = extract_frames_from_video(videoName)

    result = startAlgorithm(videoName)

    print(result)


if __name__ == '__main__':

    option = 0

    while (option != 12):
        print_menu()
        option = int(input('Elige una opción: '))

        if option == 1:
            while (option != 9):
                print_submenu1()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./VideosOriginales/akiyo.mp4"
                if option == 2:
                    videoName = "./VideosOriginales/bolas.mp4"
                if option == 3:
                    videoName = "./VideosOriginales/core.mp4"
                if option == 4:
                    videoName = "./VideosOriginales/gato.mp4"
                if option == 5:
                    videoName = "./VideosOriginales/gatos.mp4"
                if option == 6:
                    videoName = "./VideosOriginales/krav.mp4"
                if option == 7:
                    videoName = "./VideosOriginales/navidad.mp4"
                if option == 8:
                    videoName = "./VideosOriginales/nieve.mp4"

                ejecutar_cnn(videoName)

        elif option == 2:
            while (option != 9):
                print_submenu1()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./VideosComprimidos/akiyo.mp4"
                if option == 2:
                    videoName = "./VideosComprimidos/bolas.mp4"
                if option == 3:
                    videoName = "./VideosComprimidos/core.mp4"
                if option == 4:
                    videoName = "./VideosComprimidos/gato.mp4"
                if option == 5:
                    videoName = "./VideosComprimidos/gatos.mp4"
                if option == 6:
                    videoName = "./VideosComprimidos/krav.mp4"
                if option == 7:
                    videoName = "./VideosComprimidos/navidad.mp4"
                if option == 8:
                    videoName = "./VideosComprimidos/nieve.mp4"

                ejecutar_cnn(videoName)

        elif option == 3:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./InsercionClipIni/bolas.mp4"
                if option == 2:
                    videoName = "./InsercionClipIni/bus.mp4"
                if option == 3:
                    videoName = "./InsercionClipIni/claire.mp4"
                if option == 4:
                    videoName = "./InsercionClipIni/core.mp4"
                if option == 5:
                    videoName = "./InsercionClipIni/gato.mp4"
                if option == 6:
                    videoName = "./InsercionClipIni/gatos.mp4"
                if option == 7:
                    videoName = "./InsercionClipIni/krav.mp4"
                if option == 8:
                    videoName = "./InsercionClipIni/navidad.mp4"
                if option == 9:
                    videoName = "./InsercionClipIni/nieve.mp4"

                ejecutar_algoritmo(videoName)


        elif option == 4:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./InsercionClipMid/bolas.mp4"
                if option == 2:
                    videoName = "./InsercionClipMid/bus.mp4"
                if option == 3:
                    videoName = "./InsercionClipMid/claire.mp4"
                if option == 4:
                    videoName = "./InsercionClipMid/core.mp4"
                if option == 5:
                    videoName = "./InsercionClipMid/gato.mp4"
                if option == 6:
                    videoName = "./InsercionClipMid/gatos.mp4"
                if option == 7:
                    videoName = "./InsercionClipMid/krav.mp4"
                if option == 8:
                    videoName = "./InsercionClipMid/navidad.mp4"
                if option == 9:
                    videoName = "./InsercionClipMid/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 5:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./InsercionClipFin/bolas.mp4"
                if option == 2:
                    videoName = "./InsercionClipFin/bus.mp4"
                if option == 3:
                    videoName = "./InsercionClipFin/claire.mp4"
                if option == 4:
                    videoName = "./InsercionClipFin/core.mp4"
                if option == 5:
                    videoName = "./InsercionClipFin/gato.mp4"
                if option == 6:
                    videoName = "./InsercionClipFin/gatos.mp4"
                if option == 7:
                    videoName = "./InsercionClipFin/krav.mp4"
                if option == 8:
                    videoName = "./InsercionClipFin/navidad.mp4"
                if option == 9:
                    videoName = "./InsercionClipFin/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 6:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./DuplicadoClipIni/bolas.mp4"
                if option == 2:
                    videoName = "./DuplicadoClipIni/bus.mp4"
                if option == 3:
                    videoName = "./DuplicadoClipIni/claire.mp4"
                if option == 4:
                    videoName = "./DuplicadoClipIni/core.mp4"
                if option == 5:
                    videoName = "./DuplicadoClipIni/gato.mp4"
                if option == 6:
                    videoName = "./DuplicadoClipIni/gatos.mp4"
                if option == 7:
                    videoName = "./DuplicadoClipIni/krav.mp4"
                if option == 8:
                    videoName = "./DuplicadoClipIni/navidad.mp4"
                if option == 9:
                    videoName = "./DuplicadoClipIni/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 7:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./DuplicadoClipMid/bolas.mp4"
                if option == 2:
                    videoName = "./DuplicadoClipMid/bus.mp4"
                if option == 3:
                    videoName = "./DuplicadoClipMid/claire.mp4"
                if option == 4:
                    videoName = "./DuplicadoClipMid/core.mp4"
                if option == 5:
                    videoName = "./DuplicadoClipMid/gato.mp4"
                if option == 6:
                    videoName = "./DuplicadoClipMid/gatos.mp4"
                if option == 7:
                    videoName = "./DuplicadoClipMid/krav.mp4"
                if option == 8:
                    videoName = "./DuplicadoClipMid/navidad.mp4"
                if option == 9:
                    videoName = "./DuplicadoClipMid/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 8:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./DuplicadoClipFin/bolas.mp4"
                if option == 2:
                    videoName = "./DuplicadoClipFin/bus.mp4"
                if option == 3:
                    videoName = "./DuplicadoClipFin/claire.mp4"
                if option == 4:
                    videoName = "./DuplicadoClipFin/core.mp4"
                if option == 5:
                    videoName = "./DuplicadoClipFin/gato.mp4"
                if option == 6:
                    videoName = "./DuplicadoClipFin/gatos.mp4"
                if option == 7:
                    videoName = "./DuplicadoClipFin/krav.mp4"
                if option == 8:
                    videoName = "./DuplicadoClipFin/navidad.mp4"
                if option == 9:
                    videoName = "./DuplicadoClipFin/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 9:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./ShuffleClipIni/bolas.mp4"
                if option == 2:
                    videoName = "./ShuffleClipIni/bus.mp4"
                if option == 3:
                    videoName = "./ShuffleClipIni/claire.mp4"
                if option == 4:
                    videoName = "./ShuffleClipIni/core.mp4"
                if option == 5:
                    videoName = "./ShuffleClipIni/gato.mp4"
                if option == 6:
                    videoName = "./ShuffleClipIni/gatos.mp4"
                if option == 7:
                    videoName = "./ShuffleClipIni/krav.mp4"
                if option == 8:
                    videoName = "./ShuffleClipIni/navidad.mp4"
                if option == 9:
                    videoName = "./ShuffleClipIni/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 10:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./ShuffleClipMid/bolas.mp4"
                if option == 2:
                    videoName = "./ShuffleClipMid/bus.mp4"
                if option == 3:
                    videoName = "./ShuffleClipMid/claire.mp4"
                if option == 4:
                    videoName = "./ShuffleClipMid/core.mp4"
                if option == 5:
                    videoName = "./ShuffleClipMid/gato.mp4"
                if option == 6:
                    videoName = "./ShuffleClipMid/gatos.mp4"
                if option == 7:
                    videoName = "./ShuffleClipMid/krav.mp4"
                if option == 8:
                    videoName = "./ShuffleClipMid/navidad.mp4"
                if option == 9:
                    videoName = "./ShuffleClipMid/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 11:
            while (option != 10):
                print_submenu2()
                option = int(input('Elige una opción: '))
                if option == 1:
                    videoName = "./ShuffleClipFin/bolas.mp4"
                if option == 2:
                    videoName = "./ShuffleClipFin/bus.mp4"
                if option == 3:
                    videoName = "./ShuffleClipFin/claire.mp4"
                if option == 4:
                    videoName = "./ShuffleClipFin/core.mp4"
                if option == 5:
                    videoName = "./ShuffleClipFin/gato.mp4"
                if option == 6:
                    videoName = "./ShuffleClipFin/gatos.mp4"
                if option == 7:
                    videoName = "./ShuffleClipFin/krav.mp4"
                if option == 8:
                    videoName = "./ShuffleClipFin/navidad.mp4"
                if option == 9:
                    videoName = "./ShuffleClipFin/nieve.mp4"

                ejecutar_algoritmo(videoName)

        elif option == 12:
            print('Saliendo')
            exit()
        else:
            print('Opción no válida. Please enter a number between 1 and 12.')


    print("Terminado ")
