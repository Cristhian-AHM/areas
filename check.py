import cv2
import math
import imutils
import numpy as np
from imutils import perspective
from imutils import contours
from Pieces.pyimagesearch.shapedetection import ShapeDetector
from scipy.spatial import distance as dist

mm = 25.4

def determinateArea(referenceImage, fill, color, image):
	cnts = cv2.findContours(referenceImage.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	sd = ShapeDetector()
	ratio = 1
	negativeArea = 0
	# Se recorren todos los contornos encontrados.
	for c in cnts:
		# Se calcula el momento de los contornos.
		M = cv2.moments(c)
		if int(M["m00"]) == 0:
			continue
		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] / M["m00"]) * ratio)
		area = sd.detectArea(c)
		print("Areas: ", area)
		status = True
		negativeArea += area
		if status:
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			cv2.drawContours(image, [c], -1, color, fill)
		cv2.waitKey(0)
		cv2.imshow("Area", image)
	return negativeArea 

def calculate_areas():
    widthObject = float(input("Introduzca el largo del objeto: "))
    image = cv2.imread("Pieces/imagenes/pieces.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Gray" , gray)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # Se muestra la imagen
    cv2.imshow("Edges", edged)
    cv2.waitKey(0)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    sd = ShapeDetector()
    generalArea = 0
    # Se obtiene el area del objeto
    for c in cnts:
    	if cv2.contourArea(c) < 200:
    		continue
    	box = cv2.minAreaRect(c)
    	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    	box = np.array(box, dtype="int")
    	# Se ordenan los puntos del contorno en tal manera que 
    	# aparezcan en el siguiente orden: Esquina superior izquierda,
    	# esquina superior derecha, esquina inferior izquierda y esquina
    	# inferior derecha.
    	box = perspective.order_points(box)
    	#Esquina Superior Izquierda
    	cXSI = box[0, 0]
    	cYSI = box[0, 1]
    	#Esquina Superior Derecha
    	cYSD = box[1, 1]
    	#Esquina Inferior Derecha
    	cXID = box[2, 0]
    	cYID = box[2, 1]
    	#Esquina Inferior Izquierda
    	cXII = box[0, 0]
    	width2 = cXID - cXII	
    	height2 = cYID - cYSD
    	height2 += cYSI
    	width2 += cXSI
    	roi = gray[int(cYSI-5):int(height2+5), int(cXSI-5):int(width2+5)]
    cv2.imshow("ROI" , roi)
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,threshInv = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Thresh Inv", threshInv)

    negativeArea = determinateArea(threshInv, -1, (0,0,0), image)
    normalArea = determinateArea(thresh, -1, (255,255,255), image)
    newW = roi.shape[1] - 1
    ppmm = newW/(widthObject*mm)
    height = widthObject / (roi.shape[1] / roi.shape[0])
    print(roi.shape)
    print("Height: ", height)
    areaPixels = roi.shape[0] * roi.shape[1]
    areaMM = ppmm**2
    print("PPmm: ", ppmm)
    cv2.drawContours(image, [box.astype("int")], -1, (255, 255, 0), 2)
    generalArea = normalArea + negativeArea
    irregularArea = generalArea - negativeArea
    print("Area en mm: ", areaMM)
    print("Area en pixeles: ", areaPixels/areaMM)
    print("Area Total: ", generalArea/areaMM)
    print("Area Negativa: ", negativeArea/areaMM)
    print("Irregular area: ", irregularArea/areaMM)
    cv2.waitKey(0)
    cv2.destroyAllWindows()