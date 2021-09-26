import os, sys
import time
from PIL import Image
import math
import numpy
import scipy.stats.stats as scipystats
import scipy
from scipy import ndimage
from scipy import misc
#import funcionesSvsC

def statisticalValuesC(img):

    #Abrimos la imagen
    imageOriginal = img#Image.open(imageOriginalPath)
    width, height = imageOriginal.size
    imageOriginal.load()

    #start = time.time()
    r=1
    sigma=0.5
    #nucleo gaussiano
    #im = img#Image.open(imageOriginalPath)
    """Genero el nucleo gaussiano"""
    e = 2.71828183
    pi = 3.14159
    tamNucleo = r * 2 + 1
    #nucleo = [[0 for x in xrange(tamNucleo)] for y in xrange(tamNucleo)]
    nucleo = numpy.empty((tamNucleo,tamNucleo), dtype=numpy.double)
    suma = 0
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            g = (1 / (2 * pi * (sigma ** 2))) * e ** -((float(i ** 2 + j ** 2)) / (2 * (sigma ** 2)))
            nucleo[i + r][j + r] = g
            suma = suma + g
    #Normalizo
    for i in range(tamNucleo):
        for j in range(tamNucleo):
            nucleo[i][j] = nucleo[i][j] / suma
    #print nucleo

    original = numpy.asarray(img)#misc.imread(imageOriginalPath)

    original = numpy.swapaxes(original, 0, 2)
    original = numpy.swapaxes(original, 1, 2)

    col_corr = numpy.empty((width), dtype=numpy.double)
    row_corr = numpy.empty((height), dtype=numpy.double)

    gaussian = numpy.zeros((3, height, width), dtype=numpy.uint8)
    #gaussian2 = numpy.empty((3, height, width), dtype=numpy.float32)
    noise = numpy.empty((3, height, width), dtype=numpy.int32)

    noiseRowR = numpy.zeros((width), dtype=numpy.double)
    noiseRowG = numpy.zeros((width), dtype=numpy.double)
    noiseRowB = numpy.zeros((width), dtype=numpy.double)
    rowR = numpy.zeros((width), dtype=numpy.double)
    rowG = numpy.zeros((width), dtype=numpy.double)
    rowB = numpy.zeros((width), dtype=numpy.double)
    noiseColR = numpy.zeros((height), dtype=numpy.double)
    noiseColG = numpy.zeros((height), dtype=numpy.double)
    noiseColB = numpy.zeros((height), dtype=numpy.double)
    vectorAuxR = numpy.zeros((height), dtype=numpy.double)
    vectorAuxG = numpy.zeros((height), dtype=numpy.double)
    vectorAuxB = numpy.zeros((height), dtype=numpy.double)
    totalMedia = numpy.empty((1), dtype=numpy.double)
    #Funcion C
    #funcionesSvsC.gaussAndCorrelation(original,nucleo,gaussian,col_corr,row_corr,noise,noiseRowR,noiseRowG,noiseRowB,rowR,rowG,rowB,noiseColR,noiseColG,noiseColB,vectorAuxR,vectorAuxG,vectorAuxB,totalMedia)

    #funcionesC.gauss(original,nucleo,gaussian)

    #modaRow = moda(row_corr)
    #modaRow=0

    mediaRow = numpy.mean(row_corr)
    maxRow = max(row_corr)
    minRow = min(row_corr)
    medianaRow = numpy.median(row_corr)
    varianzaRow = numpy.var(row_corr)
    kurtosisRow = scipystats.kurtosis(row_corr)
    skewnessRow = scipystats.skew(row_corr)

    #Para columnas
    mediaCol = numpy.mean(col_corr)
    #mediaCol2 = media(col_corr)
    #print "Media: " + str(mediaCol==mediaCol2)
    maxCol = max(col_corr)
    minCol = min(col_corr)
    #medianaCol2 = mediana(col_corr)
    medianaCol = numpy.median(col_corr)
    #print "Mediana: " + str(medianaCol==medianaCol2)
    #modaCol = moda(col_corr)
    modaCol=0
    varianzaCol = numpy.var(col_corr)

    #desvTipicaCol = desviacionTipicaV(col_corr)
    kurtosisCol = scipystats.kurtosis(col_corr)
    skewnessCol = scipystats.skew(col_corr)
    #####
    ratio = mediaRow / mediaCol
    #####
    ##########
    returnArr = numpy.array([mediaCol, medianaCol,  maxCol, minCol, varianzaCol, kurtosisCol, skewnessCol, mediaRow, medianaRow, maxRow, minRow, varianzaRow, kurtosisRow, skewnessRow,ratio,totalMedia[0]])
    whereAreInf = numpy.isinf(returnArr)
    returnArr[whereAreInf] = 0.0
    whereAreInf = numpy.isnan(returnArr)
    returnArr[whereAreInf] = 0.0
    ##########
    return returnArr.tolist()#[mediaCol, medianaCol,  maxCol, minCol, varianzaCol, kurtosisCol, skewnessCol, mediaRow, medianaRow, maxRow, minRow, varianzaRow, kurtosisRow, skewnessRow,ratio,totalMedia[0]]

