# -*- coding: utf-8 -*-

"Algoritmos proyecto - Kevin Alejandro Hernández"


"Librerias"
import cv2

import numpy as np

import skimage.measure as sk

from scipy import ndimage

from scipy.optimize import curve_fit

from skimage import draw

import math

import os

import sys


'''Funcion para el correcto funcionamiento
   al procesar las imagenes de la interfaz,  
   ademas de facilitar la carga de otros 
   recursos necesarios'''
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


"Recuento ordenado de valores (Ordered count of values)"

def OCV(imagen):

    lista = np.reshape(imagen, imagen.shape[0] * imagen.shape[1])
    
    valores = np.array(list(range(max(lista) + 1)))
    
    recuentos = []
    
    while lista.any():
    
        for valor in valores:
            
            indices = []
            
            indices = np.where(lista == valor)
                    
            recuentos.append(np.size(indices))
                    
            lista = np.delete(lista, indices)
        
        return recuentos
    

"Adaptive Gamma Correction Weighting Distribution"

def AGCWD(imagen, alfa, t):
    
    "Reducir la intencidad de los pixeles cercanos a la mama"
    
    k = 0
    
    while k < imagen.shape[0]:
        
        fila = np.nonzero(imagen[k][:])[0]
        
        if fila.any():
        
            if imagen[k][fila[0]] <= imagen[k][fila[-1]]: t = imagen[k][fila[0]]
                
            else: t = imagen[k][fila[-1]]
            
            if t < 10: 
                
                imagen = imagen - t * np.ones(imagen.shape)
            
                imagen[imagen < 0] = 0
    
                break
        
        k += 1
        

    "Aplicar agcwd"  
    
    nl = np.array(OCV(imagen.astype(np.uint8)))
    
    f = nl / sum(nl)
    
    fw = max(f) * pow((f - min(f)) / (max(f) - min(f)), alfa)
    
    Fw = np.cumsum(f) / sum(fw)
    
    gama = 1 - Fw
    
    lmax = np.max(imagen)
    
    imagen_agcwd = np.zeros(imagen.shape)
    
    for i in range(imagen.shape[0]):
        
        for j in range(imagen.shape[1]):
            
            l = int(imagen[i][j])
            
            imagen_agcwd[i][j] = lmax * pow(l / lmax, gama[l])
    
    return imagen_agcwd


"Eliminación de artefactos"

def remosion_artefactos(imagen):

    "Convertir a escala de grices"
    
    if len(imagen.shape) > 2: imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    
    "Filtrar imagen"
    
    imagen = cv2.medianBlur(imagen, 3)
    
    
    "Ralzar contraste mama-fondo"
            
    alfa = 0.8
    
    umbral_t = 10
    
    imagen_agcwd = AGCWD(imagen, alfa, umbral_t)
    
    
    "Segmentar la mama"
    
    umbral = 0.08
    
    _, imagen_binaria = cv2.threshold(imagen_agcwd, umbral * 255, 255, cv2.THRESH_BINARY)

    
    "Eliminar franjas"
    
    frac = 15
                    
    ancho = int(imagen.shape[1] / frac)
    
    franja = np.zeros(imagen.shape) 
    
    franja[ancho:imagen.shape[0]-ancho, ancho:imagen.shape[1]-ancho] = 1
    
    imagen_binaria = imagen_binaria * franja

    
    "Seleccionar la mama"
    
    propiedades = sk.regionprops(sk.label(imagen_binaria))
    
    Area =[]

    for objeto in propiedades: Area.append(objeto.area)    
    
    ind = Area.index(max(Area))
        
    mascara = np.zeros(imagen.shape)
    
    for coordenadas in propiedades[ind].coords: mascara[coordenadas[0]][coordenadas[1]] = 1
            
    imagen_mama = imagen * mascara

    
    "Recortar mama"
    
    rectangulo = list(propiedades[ind].bbox)
    
    imagen_mama = imagen_mama[rectangulo[0]:rectangulo[2], rectangulo[1]:rectangulo[3]].astype(np.uint8)

    
    return imagen_mama


"Localizar el musculo pectoral"

def localizar_pectoral(imagen):
      
    "Reflejar imagen"
    
    ceros_der = len(np.where(imagen[0:imagen.shape[0], int(imagen.shape[1] / 2):imagen.shape[1]] == 0)[0]) 
  
    ceros_izq = len(np.where(imagen[0:imagen.shape[0], 0:int(imagen.shape[1] / 2)] == 0)[0])  
  
    if ceros_izq > ceros_der: imagen = cv2.flip(imagen, 1)
    
    
    "Eliminar el fondo y realza el pectoral"
     
    fondo = np.tile(np.linspace(0, imagen.max(), imagen.shape[1]), (imagen.shape[0], 1)) + np.tile(np.linspace(0, imagen.max(), imagen.shape[0]), (imagen.shape[1], 1)).T
    
    fondo[fondo > 255] = 255
    
    imagen_pectoral = imagen - fondo
    
    imagen_pectoral[imagen_pectoral < 0] = 0
    
    imagen_pectoral = imagen_pectoral.astype(np.uint8)
    
    return imagen_pectoral, imagen


"K-means"

def Kmeans(imagen, k): 
    
    valor_pixeles = np.float32(imagen.flatten())
    
    criterio_parada = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, etiquetas, _ = cv2.kmeans(valor_pixeles, k, None, criterio_parada, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, recuento = np.unique(etiquetas, return_counts = True)
    
    if recuento[1] > recuento[0]: etiquetas = 1 - etiquetas
    
    mascara_pectoral = etiquetas.reshape(imagen.shape).astype(np.uint8)
    
    return mascara_pectoral


"Crea un poligono dado los vertices"

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    
    mask = np.zeros(shape, dtype=np.bool)
    
    mask[fill_row_coords, fill_col_coords] = True
    
    return mask


"Corrección del contorno del pectoral"

def correccion_contorno(mascara):
    
    mascara = cv2.medianBlur(mascara, 5)
    
    "Obtener el contorno"
    
    contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    imagen_contorno = np.zeros(mascara.shape)
    
    cv2.drawContours(imagen_contorno, contornos, -1, (255,255,255), 1)

    "Filtrar el contorno"

    linea_horizontal = cv2.erode(imagen_contorno, np.ones((1, 10), np.uint8))
    
    linea_vertical = cv2.erode(imagen_contorno, np.ones((10, 1), np.uint8))
    
    lineas = linea_horizontal + linea_vertical + 1 
    
    lineas[lineas != 1] = 0
    
    imagen_contorno = (imagen_contorno * lineas).astype(np.uint8)
    
    "Convertir el contorno en par ordenado (x,y)"
    
    imagen_contorno_labeled = sk.label(imagen_contorno)

    propiedades = sk.regionprops(imagen_contorno_labeled)
    
    cont = 0
    
    coordenadas_pxl = []
    
    for i in range(len(propiedades)):
    
        if len(propiedades[i].coords) >= 30:  # Matlab 50
            
            if cont > 0: coordenadas_pxl = np.concatenate((coordenadas_pxl, propiedades[i].coords))
            
            else: coordenadas_pxl = propiedades[i].coords
                
            cont += 1
    
    if len(coordenadas_pxl) != 0:
        
        filas = coordenadas_pxl[:, 0]
        
        columnas = coordenadas_pxl[:, 1]
        
        _, indice = np.unique(filas, return_index = True)
        
        x = filas[indice]
        
        columnas_auxiliar = columnas[indice]
        
        if x[0] != 0:
            
            x = np.concatenate(([0], x))
            
            columnas_auxiliar = np.concatenate(([columnas_auxiliar[0]], columnas_auxiliar))
        
        "Polinomio de orden 3"
        
        def poly3(xVals, a, b, c, d):
    
            return a*pow(xVals, 3) + b*pow(xVals, 2) + c*xVals + d
        
        modelo,_ = curve_fit(poly3, x, columnas_auxiliar)

        y = poly3(x, *modelo)
        
        x = np.concatenate((0, x, x[-1]), axis = None)
        
        y = np.concatenate((0, y, 0), axis = None)
        
        "Aproxima el contorno a un polinomio de orden 3"
        
        mascara_corregida = poly2mask(x, y, mascara.shape).astype(np.uint8)*2
        
    else: mascara_corregida = np.zeros(mascara.shape)
    
    mascara_corregida[mascara_corregida == 0] = 1
    
    mascara_corregida[mascara_corregida == 2] = 0
 
    return  mascara_corregida


"Remoción del musculo pectoral"

def remocion_pectoral(imagen):
    
    imagen_pectoral_fondo, imagen2 = localizar_pectoral(imagen)
    
    mascara_pectoral = Kmeans(imagen_pectoral_fondo, 2)
    
    if len(np.where(mascara_pectoral != 0)[0]) != 0:
        
        "Corregir mascara del pectoral"
        
        mascara_pectoral = ndimage.binary_fill_holes(mascara_pectoral).astype(np.uint8)
        
        mascara_pectoral = cv2.dilate(mascara_pectoral, np.ones((5, 5), np.uint8))
        
        mascara_corregida = correccion_contorno(mascara_pectoral)
        
    else: mascara_corregida = mascara_pectoral
     
    imagen_pectoral = imagen2 * mascara_corregida 
    
    return imagen_pectoral
    

"Deteccion de microcalcificaciones"

def deteccion_mc(imagen, modelo):
    
    rois = []
    
    coordenadas = []
    
    "Ventanea la imagen"
    
    altura = 100
    
    ancho = 100
    
    for i in range(0, imagen.shape[0], altura):
        
        for j in range(0, imagen.shape[1], ancho):
            
            "Extrae el ROI"
            
            roi = imagen[i:i + altura, j:j + ancho]
    
            "Valida el ROI"
    
            if roi.shape[0] == altura and roi.shape[1] == ancho and roi.mean() != 0:
                
                "Clasifica el ROI"
                
                roiB = roi.reshape(-1, altura, ancho, 1)

                prediccion = np.argmax(modelo.predict(roiB), axis = -1)
                
                if prediccion[0] == 1: 
                    
                    rois.append(roi)
                    
                    coordenadas.append([i, j])
                    
    return rois, coordenadas


"Localizacion de microcalcificaciones"

def localizacion_mc(rois, coordenadas, imagen):
    
    '''Eliminar rois con mc detectadas
       anteriormente'''   
    path = resource_path('C:/Users/gomez/Documents/Tesis/algoritmos/componentes/rois_mc/region')
    
    dir_region = path
    
    path = resource_path('C:/Users/gomez/Documents/Tesis/algoritmos/componentes/rois_mc/mascara')
    
    dir_mascara = path
    
    for file in os.scandir(dir_region): os.remove(file.path)
    
    for file in os.scandir(dir_mascara): os.remove(file.path)
    
    
    res = 0.05
    
    MeanPixelIntensityValue = 40
    
    Lmax = int(1/res)
    
    Amin = 5
    
    Amax = int(math.pi * pow(Lmax/2, 2))
    
    centroides = []
    
    "Si hay rois"
    
    if rois != []:
    
        "Realce de calcificaciones"
        
        centroides = []
        
        contador = 0
        
        for indice, roi in enumerate(rois):         
            
            "Remueve el fondo"
            
            copia_roi = roi.astype(np.float)
            
            copia_roi[copia_roi == 0] = float('nan')
            
            pixel_media = np.nanmean(copia_roi)
            
            fondo = pixel_media * np.ones((100, 100))
            
            roiB = roi - fondo
            
            roiB[roiB < 0] = 0
            
            roiB = roiB.astype(np.uint8)
            
            
            "Binarizar y obtiene propiedades de los objetos"
            
            retangulo, binario = cv2.threshold(roiB, roiB.max()/2, 1, cv2.THRESH_BINARY)
            
            binario_label = sk.label(binario)
            
            propiedades = sk.regionprops(binario_label, roiB)
            
            
            "Verifica si el objeto es microcalcificacion"
            
            if len(propiedades) != 0:
                            
                for n in range(len(propiedades)):
                    
                    if (propiedades[n].major_axis_length < Lmax)and(propiedades[n].area > Amin)and(propiedades[n].area < Amax)and(propiedades[n].mean_intensity > MeanPixelIntensityValue):
                        
                        y, x = propiedades[n].centroid
                                    
                        x = int(x) + coordenadas[indice][1]
                        
                        y = int(y) + coordenadas[indice][0]
                        
                        centroides.append((x, y))
                        
                        
                        '''Guarda los rois y binarios con mc en un carpeta
                           para una posterior busqueda de rois
                           similares'''
                        contador += 1
                        
                        imagen_roi = Image.fromarray(roi)
                        
                        binario[binario == 1] = 255
                        
                        imagen_binario = Image.fromarray(binario)
        
                        imagen_roi.save('C:/Users/gomez/Documents/Tesis/algoritmos/componentes/rois_mc/region/roi'+ str(contador)+'.png')
                        
                        imagen_binario.save('C:/Users/gomez/Documents/Tesis/algoritmos/componentes/rois_mc/mascara/roi'+ str(contador)+'.png')
    
    
    "Subraya las mc"

    for mcc in centroides: imagen_localizada = cv2.circle(imagen, mcc, 15, (0, 255, 0), 2)
    
    else:
    
        imagen_localizada = imagen
    
    
    return imagen_localizada