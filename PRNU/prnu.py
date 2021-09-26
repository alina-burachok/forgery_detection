#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function


from PIL import Image, ImageChops
import time
import sys, pywt, numpy, optparse, os
from cropImage import *
from constants import *
import waveletsC

#from skimage import restoration, img_as_float, color, data
import cv2

#import waveletsC2 as waveletsC

if os.name == 'nt':
    from time import clock  # noqa
else:
    from time import time as clock  # noqa

import math, time, scipy.io


def lawmlNSmartC(X):
    """ Return the denoised image using NxN LAW-ML.
    Estimate the variance of original noise-free image for each wavelet
    coefficients using the MAP estimation for 4 sizes of square NxN
    neighborhood for N=[3,5,7,9]
    This function makes the same as lawmlN in language C quicker. """
    (m0,m1) = X.shape
    S = numpy.zeros((m0,m1))
    T = numpy.zeros((m0,m1))
    M1 = numpy.zeros((m0,m1))
    X = X.astype(numpy.float64)
    sig0 = SIGMA*SIGMA
    if MULTIPLE_NEIGHBOR:
        #X image - S and T aux matrix - M1 denoised image - sig0 SIGMA*SIGMA - 
        #1 Multiple neighbor 0 NO Multiple neighbor
        waveletsC.lawmlNC(X,S,T,M1,sig0,1) 
    else:
        waveletsC.lawmlNC(X,S,T,M1,sig0,0)
    return M1

###########################################################################
def wam9(C, nmom=9):
    "Calculate 9 WAM features from the wavelet component C."
    ##M0 = C - lawmlN(C) ##Wavelets de Joss en Python
    "Calculate 9 WAM features from the wavelet component C."
    M0 = C - lawmlNSmartC(C)#fastLawmlN(C)#
    R = numpy.mean(M0)
    M = numpy.abs(M0.flatten().astype(float) - R)
    MP = [M/M.size]
    for i in range(1, nmom):
        MP.append(M*MP[i-1])
    #return [R] + [numpy.sum(M ** k) / M.size for k in range(2, nmom + 1)]
    return [numpy.sum(M) for M in MP]#[numpy.sum(M ** k)/ M.size for k in range(1, nmom + 1)]

    #M0 = C - lawmlNSmartC(C) ##Wavelets David en C numpy.array(C - lawmlNSmartC(C), dtype='float128')
    #R = numpy.mean(M0)
    #M = M0.flatten().astype(float) - R #M0.flatten().astype(dtype='float128') - R#
    ####
    #[R] + [numpy.sum(abs(M) ** k) / M.size for k in range(2, nmom + 1)] David 2º opcion (no buena [R] puede ser negativo
    #[numpy.sum(abs(M) ** k) / M.size for k in range(1, nmom + 1)] David 1º opcion (creo que es la buena)
    #return [R] + [numpy.sum(M ** k) / M.size for k in range(2, nmom + 1)] Original Joselin
    #return [numpy.sum(abs(M) ** k) / M.size for k in range(1, nmom + 1)]

def wamNxN(I, nmom=9):
    "Calculate the WAM(N)-27 from the pixmap array I."
    (L, H) = pywt.dwt2(I,WAVELET)
    return reduce( list.__add__, [ wam9(C, nmom)for C in H])

def image2array(image):
    """PIL Image to NumPy array"""
    arr = numpy.array(image)
    #im3 = numpy.rollaxis(arr, axis=-1).astype(numpy.float32)
    arr2 = arr.swapaxes(0, 2).swapaxes(1, 2)
    #result = im3 - arr2
    #print result.max(), result.min()
    #return arr.swapaxes(0, 2).swapaxes(1, 2).astype(numpy.float32)
    return arr2#im3

def array2image(arr, mode='RGB'):
    """NumPy array to PIL Image"""
    arr = arr.swapaxes(1, 2).swapaxes(0, 2)
    arr[arr < 0] = 0
    arr[arr > 255] = 255
    arr = numpy.fix(arr).astype(numpy.uint8)
    return Image.fromarray(arr)
    
def loadImage(path, mode=None, size=None):
    #TODO: Meter en la función de cargado la opción de recorte
    """Load image"""
    im = Image.open(path)

    if im.mode not in ('L', 'P', 'RGB', 'CMYK', 'RGBA'):
        raise TypeError("Image mode must be 'L', 'P', 'RGB' or 'CMYK' or 'RGBA'")

    if mode is not None:
        if mode == 'P':
            raise ValueError("Mode must be 'L', 'RGB' or 'CMYK'")
        im = im.convert(mode)
    elif im.mode == 'P':
        im = im.convert('RGB')

    if size is not None and im.size != size:
        im = im.resize(size, Image.ANTIALIAS)
    return im
    
def extractNoise(image,wavelet, level, mode='sym'):
    """Extracts noise signal that is locally Gaussian N(0,sigma^2)"""
    #start_Inicial = time.time()
    imageData = image2array(image)
    outputData = []
    
    #Procesando canales de colores
    #start_bucle_1 = time.time()
    for n, imageBand in enumerate(imageData):
        ##print "*", n
        # calculando la descomposici?n wavelet 8-tap daubechies QMF
        imageCoeffs = pywt.wavedec2(imageBand, wavelet, mode, level)
        outputBandCoeffs = [imageCoeffs[0]]  # cA
        # eliminando el coeficiente que contiene toda la informacion L y dejando s?lo los detalles
        del imageCoeffs[0]
        # para cada banda de wavelet
        #start_bucle_2 = time.time()
        for n, imageDetails in enumerate(imageCoeffs):
            ##print "**", n
            #print imageDetails
            resDetails = []
            # para cada subbanda V, H, D
            #start_bucle_3 = time.time()
            for n, imageDetail in enumerate(imageDetails):
                ##print "*** detail antes ", n, " - ", imageDetail.size
                # #print imageDetail
                # estimando la varianza local
                resDetail = lawmlNSmartC(imageDetail)#fastLawmlN(imageDetail)#
                resDetails.append(resDetail)
                # #print "*** detail despues", " - ", imageDetail.size
                # #print resDetail
                # #print " "
            #imageCoeffs[n] = None
            #print "Fin bucle 3: ", str(time.time() - start_bucle_3)
            outputBandCoeffs.append(resDetails)
        # reconstruyendo la imagen con los nuevos coeficientes wavelet
        #print "Fin bucle 2: ", str(time.time() - start_bucle_2)
        newBand = pywt.waverec2(outputBandCoeffs, wavelet, mode)
        outputData.append(newBand)
    #print "Fin bucle 1: ", str(time.time() -start_bucle_1)
    outputData = numpy.array(outputData)
    #print "Final de metodo: ", str(time.time() - start_Inicial)
    return outputData
    
#Lo hace solo en la carpeta del path
def getAverageNoise2(path, crop, wavelet, level, mode):
    imNum = 0
    for f in os.listdir(path):
        if os.path.splitext(f)[1].lower() in EXTS:
            im = os.path.join(path, f)
            if crop:
                image = cropImage(path, f, CROP_X_SIZE, CROP_Y_SIZE, CROPCENT_REQUIRED)
            else:
                image = loadImage(im)
            x,y = image.size
            if imNum is 0:
                averageIms = [[[0.0 for i in range(x)] for i in range(y)] for i in range(3)]
                averageIms = numpy.array(averageIms)
            #TODO: Antes de llegar aqui validar si todas la imagenes de la carpeta sean del mismo tamaño
            if x==CROP_X_SIZE and y==CROP_Y_SIZE:
                imDenoisedData = extractNoise(image, wavelet, level, mode)
                averageIms += imDenoisedData
                imNum += 1
    if imNum >=1:
        averageIms /= imNum
        averageIms = zeroMean(averageIms)
        averageIms = augmentGreenChanel(averageIms)

        if FEATURES_REQUIRED:
            feats = extractFeatures(averageIms, path)
            file = open(path + os.sep + FEATURES_PREFIX +  buildFileNameDescription() + ".txt",'a')
            file.write(feats)
            file.close()
    return averageIms

#solo en subcarpetas
def getAverageNoise(path, crop, wavelet, level, mode):
    imNum = 0
    for dir, subdirs, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1].lower() in EXTS:
                im = os.path.join(dir, name)
                if crop:
                    image = cropImage(dir, name, CROP_X_SIZE, CROP_Y_SIZE, CROPCENT_REQUIRED)
                else:
                    image = loadImage(im)
                x,y = image.size
                if imNum is 0:
                    averageIms = [[[0.0 for i in range(x)] for i in range(y)] for i in range(3)]
                    averageIms = numpy.array(averageIms)
                #TODO: Antes de llegar aqui validar si todas la imagenes de la carpeta sean del mismo tamaño
                if x==CROP_X_SIZE and y==CROP_Y_SIZE:
                    imDenoisedData = extractNoise(image, wavelet, level, mode)
                    averageIms += imDenoisedData
                    imNum += 1
    if imNum >=1:
        print ("Hola1")
        averageIms /= imNum
        averageIms = zeroMean(averageIms)
        averageIms = augmentGreenChanel(averageIms)

    if FEATURES_REQUIRED:
        print ("Hola2", path)
        feats = extractFeatures(averageIms, path)
        file = open(path + os.sep + FEATURES_PREFIX +  buildFileNameDescription() + ".txt",'a')
        file.write(feats)
        file.close()
    return averageIms

# Sobre la carpeta y las subcarpetas
def extractNoiseFromDir(path, crop, wavelet, level, mode):

    for path, subdirs, files in os.walk(path):
        for name in files:
            feats = ""
            if FEATURES_REQUIRED:
                file = open(path + os.sep + FEATURES_PREFIX +  buildFileNameDescription() + ".txt",'a')
            if os.path.splitext(name)[1].lower() in EXTS:
                print ("Extracting  noise from: " + os.path.join(path, name))
                noiseData = getNoise(path, name, crop, wavelet, level, mode)
                if SAVEIMGS_REQUIRED:
                    array2image(noiseData).save(path + os.sep + NOISE_PREFIX + name)
                if FEATURES_REQUIRED:
                        feats = extractFeatures(noiseData, path) + "\n"
                        file.write(feats)
                        file.close()

# Lo hace solamente sobre la carpeta de path
def extractNoiseFromDir2(path, crop, wavelet, level, mode):
    feats = ""
    if FEATURES_REQUIRED:
        file = open(path + os.sep + FEATURES_PREFIX +  buildFileNameDescription() + ".txt",'a')
    for f in os.listdir(path):
        if os.path.splitext(f)[1].lower() in EXTS:
            print (f)
            noiseData = getNoise(path, f, crop, wavelet, level, mode)
            if SAVEIMGS_REQUIRED:
                array2image(noiseData).save(path + os.sep + NOISE_PREFIX + f)
            if FEATURES_REQUIRED:
                feats = extractFeatures(noiseData, path) + "\n"
                file.write(feats)
    if FEATURES_REQUIRED:
        file.close()

def getNoise(path, imageName, crop, wavelet, level, mode):
    if crop:
        image = cropImage(path, imageName, CROP_X_SIZE, CROP_Y_SIZE, CROPCENT_REQUIRED)
    else:
        image = loadImage(os.path.join(path,imageName))
        #image = loadImage(path + os.sep + imageName)
    imDenoisedData = extractNoise(image, wavelet, level, mode)
    if SAVEIMGS_REQUIRED:
        array2image(imDenoisedData).save(path + os.sep + DENOISED_PREFIX + imageName)
    #noise = ImageChops.difference(image, imDenoised)
    noiseData = substract(image2array(image), imDenoisedData)
    if ZEROMEAN_REQUIRED:
        noiseData = zeroMean(noiseData)
    if AUGGREEN_REQUIRED:
        noiseData = augmentGreenChanel(noiseData)
    return noiseData

def zeroMean(imageData):
    (z, y, x) = imageData.shape
    
    for c in xrange(z): #by each color band    
        "zero mean cols"
        for i in xrange(x):
            ri = numpy.average(imageData[c, :, i]) #compute row average                
            for j in xrange(y):
                imageData[c, j, i] = imageData[c, j, i] - ri
            
        "zero mean rows"
        for j in xrange(y):
                cj = numpy.average(imageData[c, j, :])#compute column average
                for i in xrange(x):
                    imageData[c, j, i] = max(0, imageData[c, j, i] - cj)
    return imageData

def augmentGreenChanel(imageData):
    imageData[0,:,:]*= 0.3
    imageData[1,:,:]*= 0.6
    imageData[2,:,:]*= 0.1
    return imageData


def substract(imageData1, imageData2):
    try:
        (z, y, x) = imageData1.shape
        imageData2 = imageData2[:,0:y,0:x]
        #start = time.time()
        #noise = numpy.clip(imageData1 - imageData2,0,None)
        noise = imageData1 - imageData2
        # numpy.save('/Users/esteban/Desktop/noise.numpy',noise)
        # -*- ToDo Legacy Code -*-
        # print 'Resta simetrica', str(time.time()-start)
        # print noise.max(), noise.min()
        # print type(noise)
        # start = time.time()
        # for c in xrange(z):
        #     for i in xrange(x):
        #         for j in xrange(y):
        #             imageData1[c, j, i] = max(0, imageData1[c, j, i] - imageData2[c, j, i])
        # print 'Resta en for', str(time.time() - start)
        # print imageData1.max(), imageData1.min()
        return noise#imageData1
    except Exception as e:
        print ("Error in  substract from noise:",e)
        raise

def extractFeatures(imageData, path):
    # Para que no sea de manera manual este número
    features = []
    for n, imageBand in enumerate(imageData):
        features += wamNxN(imageBand)
    featsStr = TRAIN_CLASS_ID
    for n, f in enumerate(features):
        featsStr += str(n) + ":" + str(f) + " "
    return featsStr

"""
def extractWaveletsFeatures(imageData):
    # Para que no sea de manera manual este número
    features = []
    numberOfBands=0
    for imageBand in imageData:
        features += wamNxN(imageBand)
        numberOfBands=numberOfBands+1
    return features, numberOfBands
"""

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n: 
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: 
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """
  
    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')
        
    while True:
        ans = raw_input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print ('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

def checkImagesSize(path, crop):
    images = []
    resp = True
    for f in os.listdir(path):
        if os.path.splitext(f)[1].lower() in EXTS:
            im = Image.open(path + os.sep +f)
            x,y = im.size
            if x<CROP_X_SIZE or y<CROP_Y_SIZE:
                images.append(f)
    if len(images) > 0:
        print ("The images: \n", images, "\n are smaller than ", str(CROP_X_SIZE), str(CROP_Y_SIZE))
        resp = confirm()
    return resp

def buildFileNameDescription():
    if ZEROMEAN_REQUIRED:
        fileNameDesc = ZERO_MEAN_SUFFIX
    else:
        fileNameDesc = NOZERO_MEAN_SUFFIX
    if MULTIPLE_NEIGHBOR:
        fileNameDesc += N3579_SUFFIX
    else:
        fileNameDesc += N3_SUFFIX
    return fileNameDesc

def main():

    usage = "usage: %prog -i IMAGE -p PATH"\
            "[-a AVERAGE][-c CROP][-t TIME]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-i", "--image", dest="image", metavar="IMAGE",
                      help="image name (for extract noise)")
    parser.add_option("-p", "--path", dest="path", metavar="PATH",
                      help="images's dir's path")
    parser.add_option("-a", "--average", dest="average", action="store_true",
                       help="if average denoised required")
    parser.add_option("-c", "--crop", dest="crop", action="store_true",
                      default=False, help="Cropping images")
    parser.add_option("-t","--timeit", dest="timeit", action="store_true",
                      default=False, help="time extracting fingerprint operations")

#  Pendiente evaluar si las opciones son mejor con las constantes como archivo de configuración o como parámetros
    (options, args) = parser.parse_args()

    if options.path is None:
        options.path = DEFAULT_PATH
        if options.image is None:
            parser.print_help()
            sys.exit(-1)

    print ("CROP_X_SIZE: ", CROP_X_SIZE)
    print ("CROP_Y_SIZE: ", CROP_Y_SIZE)
    print ("FEATURES_REQUIRED: ", FEATURES_REQUIRED)
    print ("ZEROMEAN_REQUIRED: ", ZEROMEAN_REQUIRED)
    print ("AUGGREEN_REQUIRED: ", AUGGREEN_REQUIRED)
    print ("SAVEIMGS_REQUIRED: ", SAVEIMGS_REQUIRED)
    print ("MULTIPLE_NEIGHBOR: ", MULTIPLE_NEIGHBOR)
    print ("CROPCENT_REQUIRED: ", CROPCENT_REQUIRED, "\n")

    if options.average and options.path is not None:
        print ("Calculando el patrón promedio de las images en", options.path)
        if options.timeit:
            t = clock()

        if checkImagesSize(options.path, options.crop):
            promNoiseData = getAverageNoise(options.path, options.crop, WAVELET, LEVEL, MODE)
            promNoise = array2image(promNoiseData)
            promNoise.save(options.path + os.sep + PROM_NOISE_NAME)

    elif options.image is not None:
        print ("Extrayendo la el prnu de la imagen " + options.image)
        if options.timeit:
            t = clock()
        noiseData = getNoise(options.path, options.image, options.crop,  WAVELET, LEVEL, MODE)
        array2image(noiseData,'RGB' ).save(options.path + os.sep + NOISE_PREFIX + options.image)

        if FEATURES_REQUIRED:
            feats = extractFeatures(noiseData, options.path)
            file = open(options.path + os.sep + FEATURES_PREFIX +  buildFileNameDescription() + options.image + ".txt",'a')
            file.write(feats)
            file.close()

    elif options.path is not None:
        if options.timeit:
            t = clock()
        print ("Extracting the PRNU from images in: " + options.path)
        extractNoiseFromDir(options.path, options.crop, WAVELET, LEVEL, MODE)

    if options.timeit:
        print ("%.3fs" % (clock() - t))
    print ("Process finalized")

def extract_noise(path, image):
    try:
        matrix = getNoise(path,image,CROP_REQUIRED,WAVELET,LEVEL,MODE)
        print (matrix)
    except Exception as e:
        print ('Error in function extract_noise:',e)
        raise
if __name__ == '__main__':
    path = '/home/ali/Imágenes/Wallpapers'
    image = 'test.jpeg'

    cont = 0
    path = "/home/ali/Documentos/img/"
    pathNoise = "/home/ali/Documentos/noise/"

    while (cont < 20) :

        noise = getNoise(path, "imagen"+str(cont)+".jpg", CROP_REQUIRED, WAVELET, LEVEL, MODE)
        output_path = os.path.join(pathNoise, "noise" + str(cont))
        numpy.save(output_path,noise)
        # print (noise)



        # array2image(noise).save(pathNoise + "noise" + str(cont) + ".jpg")

        cont += 1

    print ("Terminado")




   # extract_noise(path,image)
    # main()

#TODO: Agregar el manejos de excepciones.
#TODO: Medir el tiempo de las funciones para ver cuales pasar a c

# def lawmlN(X):
#     """ Return the denoised image using NxN LAW-ML.
#     Estimate the variance of original noise-free image for each wavelet
#     coefficients using the MAP estimation for 4 sizes of square NxN
#     neighborhood for N=[3,5,7,9] """
#     sig0 = SIGMA*SIGMA
#     N = (5,7,9)
#     (m0,m1) = X.shape
#     S = numpy.zeros((m0,m1))
#
#     #calculando  para tamaño N=3 para tenerlo como mínimo
#     for x in xrange(m0):
#         for y in xrange(m1):
#             sub = subm(X,3,x,y)
#             S[x,y] = max(0, numpy.var(sub) - sig0 )
#
#     if MULTIPLE_NEIGHBOR:
#         #calculando  para el restode los tamaños N=5,7,9
#         T = numpy.zeros((m0,m1))
#         for n in N:
#             for x in xrange(m0):
#                 for y in xrange(m1):
#                     sub = subm(X,n,x,y)
#                     T[x,y] = max(0, numpy.var(sub) - sig0 )
#             #print "---  S:"
#             #print S
#             #print "---  T: "
#             #print T
#             S = numpy.fmin(S,T)
#         #print "___  S min: "
#         #print S
#     return X*S/(S+sig0)
