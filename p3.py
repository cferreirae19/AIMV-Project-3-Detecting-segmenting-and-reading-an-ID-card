import cv2
import numpy as np
import os
import pytesseract

# Variables globales
finalID = None
finalID_backside = None
crop = None
cutID = None
signature = None
rotation_matrix = None
vertex = [None, None, None, None]
cont_frame = 0
control_frame = False
messageID = False
redo = False
angle = 0
rect = None
rectaux = None
xmax, xmin, ymax, ymin = 0, 0, 0, 0
turnedImage = None
heightRectangle, widthRectangle = 0, 0
capture = cv2.VideoCapture()

def detectionID(frame):
    global messageID, redo, cont_frame, control_frame, angle, rect, heightRectangle, widthRectangle, vertex
    # Convertir a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.blur(frame_gray, (3, 3))
    edges = cv2.Canny(edges, 100, 200)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
    boundRect = [cv2.boundingRect(cnt) for cnt in contours_poly]

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i], False)
        if area >= 10000:
            redo = False
            if not messageID:
                print("DNI detectado, por favor enfocar lo mejor posible.")
            messageID = True

            cv2.rectangle(frame, (boundRect[i][0], boundRect[i][1]),
                          (boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]), (0, 0, 255), 2)

            cont_frame += 1
            print(f"cont_frame: {cont_frame}")
            if cont_frame >= 30:  # 1 segundo (30) detectando DNI para enfocar bien (30 frames por segundo)
                control_frame = True
                #print(boundRect[i])
                rect = cv2.minAreaRect(contours[i])
                #print(f"rect: {rect}")
                vertex = cv2.boxPoints(rect)
                vertex = np.intp(vertex)
                angle = rect[2]  # Ángulo del DNI (rectángulo verde)
                heightRectangle = boundRect[i][3]
                widthRectangle = boundRect[i][2]

        else:
            redo = True
            break

    return frame

def IDrescale(turnedImage):
    vertex_aux = [None, None, None, None]
    xmax, xmin, ymax, ymin = 0, 0, 0, 0

    gray_turnedImage = cv2.cvtColor(turnedImage, cv2.COLOR_BGR2GRAY)
    edgesaux = cv2.blur(gray_turnedImage, (3, 3))
    edgesaux = cv2.Canny(edgesaux, 20, 275)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(edgesaux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
    boundRect = [cv2.boundingRect(cnt) for cnt in contours_poly]


    for i in range(len(contours)):
        area = cv2.contourArea(contours[i], False)
        if area >= 10000:
            rect = cv2.minAreaRect(contours[i])
            print(f"rect: {rect}")
            vertex_aux = cv2.boxPoints(rect)
            vertex_aux = np.intp(vertex_aux)
            angle = rect[2]  # Ángulo del DNI

            xmax = vertex_aux[0][0]
            xmin = vertex_aux[0][0]
            ymax = vertex_aux[0][1]
            ymin = vertex_aux[0][1]

            # print(f"DNI Reescalado xmax: {xmax}, xmin: {xmin}, ymax: {ymax}, ymin: {ymin}")

    if(xmax==0 and xmin==0 and ymax==0 and ymin==0):
        print("No se ha detectado el DNI correctamente, por favor reinicie el programa.")
        exit()

    for i in range(4):
        if xmax <= vertex_aux[i][0]:
            xmax = vertex_aux[i][0]
        if ymax <= vertex_aux[i][1]:
            ymax = vertex_aux[i][1]
        if xmin >= vertex_aux[i][0]:
            xmin = vertex_aux[i][0]
        if ymin >= vertex_aux[i][1]:
            ymin = vertex_aux[i][1]

    definitivo_aux = turnedImage[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    definitivo_aux = cv2.resize(definitivo_aux, (425, 270))
    return definitivo_aux


def faceExtract(finalID):
    facecrop = None
    detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(finalID, cv2.COLOR_BGR2GRAY)
    dest = cv2.equalizeHist(gray)
    rect = detector.detectMultiScale(dest)

    for rc in rect:
        cv2.rectangle(finalID, (rc[0], rc[1]), (rc[0] + rc[2], rc[1] + rc[3]), (0, 0, 0), 0)
        facecrop = finalID[rc[1]:rc[1] + rc[3], rc[0]:rc[0] + rc[2]]

    cv2.imshow("Face ID", facecrop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def surnameExtract(finalID):
    surname = finalID[80:110, 165:300] # 220:345, 90:150
    surname_gray = cv2.cvtColor(surname, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/surname.jpg", surname_gray)
    cv2.imshow("Surname", surname_gray)
    #print("Lectura del surname:")
    #system_output = pytesseract.image_to_string("surname.jpg", config="--oem 3 --psm 6")
    #print(system_output)
    return surname


def nameExtract(finalID):
    name = finalID[114:132, 165:300] # 220:320, 148:183
    name_gray = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/name.jpg", name_gray)
    cv2.imshow("Name", name_gray)
    #print("Lectura del name:")
    #system_output = pytesseract.image_to_string("name.jpg", config="--oem 3 --psm 6")
    #print(system_output)
    return name


def numberExtract(finalID):
    IDnumber = finalID[40:68, 180:325]
    # IDnumber = finalID[238:261, 38:160]
    numero_gray = cv2.cvtColor(IDnumber, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/IDnumber.jpg", numero_gray)
    cv2.imshow("ID Number", numero_gray)
    #print("Lectura del número de DNI:")
    #system_output = pytesseract.image_to_string("IDnumber.jpg", config="--oem 3 --psm 6")
    #print(system_output)
    return IDnumber

def signatureExtract(finalID):
    signature = finalID[200:250, 175:300]
    signature_gray = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/signature.jpg", signature_gray)
    cv2.imshow("signature", signature_gray)
    return signature

def dueDateExtract(finalID):
    dueDate = finalID[160:180, 245:330]
    dueDate_gray = cv2.cvtColor(dueDate, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/dueDate.jpg", dueDate_gray)
    cv2.imshow("Due date", dueDate_gray)
    #print("Lectura de la fecha de caducidad:")
    #system_output = pytesseract.image_to_string("caducidad.jpg", config="--oem 3 --psm 6")
    #print(system_output)
    return dueDate

def birthdayExtract(finalID):
    birthday = finalID[135:155, 330:420]
    birthday_gray = cv2.cvtColor(birthday, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/birthday.jpg", birthday_gray)
    cv2.imshow("Birthday", birthday_gray)
    #print("Lectura de la fecha de nacimiento:")
    #system_output = pytesseract.image_to_string("nacimiento.jpg", config="--oem 3 --psm 6")
    #print(system_output)
    return birthday

def mrzExtract(finalID_backside):
    mrz = finalID_backside[155:269, 1:424]
    mrz_gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/mrz.jpg", mrz_gray)
    #print("Lectura del MRZ:")
    #system_output = pytesseract.image_to_string("mrz.jpg", config="--oem 3 --psm 6")
    #print(system_output)
    return mrz

def main():
    '''
    temp1 = cv2.imread("DNITemplateFrontal_1.jpg")
    temp1 = cv2.resize(temp1, (425, 270))
    temp2 = cv2.imread("DNITemplateFrontal_2.jpg")
    temp2 = cv2.resize(temp2, (425, 270))

    cv2.imshow("Template 1", temp1)
    cv2.imshow("Template 2", temp2)
    '''

    global rotation_matrix, control_frame, redo
    capture = cv2.VideoCapture(0)

    frame, frame_back = None, None
    while True:
        ret, frame = capture.read()
        ret, crop = capture.read()

        frame = detectionID(frame)

        cv2.namedWindow("Normal Video")
        cv2.imshow("Normal Video", frame)

        if control_frame == True:
            print("Presione cualquier tecla para continuar...")
            cv2.waitKey()
            break
        else:
            if redo == True:
                cont_frame = 0
            if cv2.waitKey(30) >= 0:
                break

    xmax = vertex[0][0]
    xmin = vertex[0][0]
    ymax = vertex[0][1]
    ymin = vertex[0][1]

    for i in range(4):
        if xmax <= vertex[i][0]:
            xmax = vertex[i][0]
        if ymax <= vertex[i][1]:
            ymax = vertex[i][1]
        if xmin >= vertex[i][0]:
            xmin = vertex[i][0]
        if ymin >= vertex[i][1]:
            ymin = vertex[i][1]

    print(f"xmax: {xmax}, xmin: {xmin}, ymax: {ymax}, ymin: {ymin}")

    cutID = crop[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    output_directory = "Imagenes"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    cv2.imwrite("Imagenes/cutID.jpg", cutID)
    cv2.imshow("IDdetect", cutID)

    center = (cutID.shape[1] - 1) / 2.0, (cutID.shape[0] - 1) / 2.0

    # ----------------------------Case ID rotated to the right----------------------------------
    if heightRectangle < widthRectangle and angle >= 60 and angle <= 90:
        print("DNI rotated to the right")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle-90, 1.0)

    # ----------------------------Case ID rotated to the left----------------------------------
    elif heightRectangle < widthRectangle and angle >= 0 and angle <= 30:
        print("DNI rotated to the left")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    else:
        print("No se ha detectado el DNI correctamente, por favor reinicie el programa.")
        exit()

    # We rotate the image using warpAffine
    print(f"angle: {angle}")
    print(f"cutID.shape[0]: {cutID.shape[0]}, cutID.shape[1]: {cutID.shape[1]}")
    turnedImage = cv2.warpAffine(cutID, rotation_matrix, (cutID.shape[1], cutID.shape[0]))
    cv2.imshow("DNI rotado en uprigth position", turnedImage)
    cv2.imwrite("Imagenes/turnedImage.jpg", turnedImage)

    # We remove unwanted external edges from the rotated ID and resize to handle a known image size
    # If the ID is at a correct angle from the beginning (no garbage edges in the image) we only resize, no need to cut more image
    if (angle <= 2 and angle >= 0) or (angle >= 88 and angle <= 90):
        finalID = turnedImage
        finalID = cv2.resize(finalID, (425, 270)) # 365, 575
    # If these "garbage" edges exist we call IDrescale() and cut off the excess edges
    else:
        finalID = IDrescale(turnedImage)
    cv2.imshow("DNI", finalID)
    cv2.imwrite("Imagenes/finalID.jpg", finalID)
    # We get all the data from the front part of the ID in different images
    faceExtract(finalID)
    cv2.imshow("ID Number", numberExtract(finalID))
    cv2.imshow("Name", nameExtract(finalID))
    cv2.imshow("Surname", surnameExtract(finalID))
    cv2.imshow("signature", signatureExtract(finalID))
    cv2.imshow("Due date", dueDateExtract(finalID))
    cv2.imshow("Birthday date", birthdayExtract(finalID))
    cv2.waitKey(0)


    # REVERSO DEL DNI _________________________________________________________________
    control_frame = False
    redo = False
    messageID = False

    capture = cv2.VideoCapture(0)

    frame, frame_back = None, None
    while True:
        ret, frame_back = capture.read()
        ret, crop = capture.read()

        frame_back = detectionID(frame_back)

        cv2.namedWindow("Normal Video")
        cv2.imshow("Normal Video", frame_back)

        if control_frame == True:
            print("Presione cualquier tecla para continuar...")
            cv2.waitKey()
            break
        else:
            if redo == True:
                cont_frame = 0
            if cv2.waitKey(30) >= 0:
                break

    # We use the positions of the vertices of the ID (green rectangle) to segment it from the rest of the frame:
    xmax = vertex[0][0]
    xmin = vertex[0][0]
    ymax = vertex[0][1]
    ymin = vertex[0][1]

    for i in range(4):
        # UNCOMMENT TO SEE GREEN RECTANGLE
        # cv2.line(crop, vertex[i], vertex[(i + 1) % 4], (0, 255, 0), 3)

        if xmax <= vertex[i][0]:
            xmax = vertex[i][0]
        if ymax <= vertex[i][1]:
            ymax = vertex[i][1]
        if xmin >= vertex[i][0]:
            xmin = vertex[i][0]
        if ymin >= vertex[i][1]:
            ymin = vertex[i][1]

    print(f"xmax: {xmax}, xmin: {xmin}, ymax: {ymax}, ymin: {ymin}")

    # cv2.imshow("DNI capture", crop) # green square

    # We rotate in a new image the ID already segmented from the rest of the image:
    cutID = crop[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    cv2.imwrite("Imagenes/cutIDBack.jpg", cutID)
    # cutID = crop.crop((left, top, right, bottom))
    cv2.imshow("IDdetect", cutID)

    center = (cutID.shape[1] - 1) / 2.0, (cutID.shape[0] - 1) / 2.0  # We find the center of the ID

    # ----------------------------Case ID rotated to the right----------------------------------
    if heightRectangle < widthRectangle and angle >= 60 and angle <= 90:
        print("DNI rotated to the right")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)

    # ----------------------------Case ID rotated to the left----------------------------------
    elif heightRectangle < widthRectangle and angle >= 0 and angle <= 30:
        print("DNI rotated to the left")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    else:
        print("No se ha detectado el DNI correctamente, por favor reinicie el programa.")
        exit()

    # We rotate the image using warpAffine
    print(f"angle: {angle}")
    print(f"cutID.shape[0]: {cutID.shape[0]}, cutID.shape[1]: {cutID.shape[1]}")
    turnedImage = cv2.warpAffine(cutID, rotation_matrix, (cutID.shape[1], cutID.shape[0]))
    cv2.imshow("DNI rotado en uprigth position", turnedImage)
    cv2.imwrite("Imagenes/turnedImage_back.jpg", turnedImage)

    # We remove unwanted external edges from the rotated ID and resize to handle a known image size
    # If the ID is at a correct angle from the beginning (no garbage edges in the image) we only resize, no need to cut more image
    if (angle <= 2 and angle >= 0) or (angle >= 88 and angle <= 90):
        finalID = turnedImage
        finalID = cv2.resize(finalID, (425, 270))  # 365, 575
    # If these "garbage" edges exist we call IDrescale() and cut off the excess edges
    else:
        finalID = IDrescale(turnedImage)
    cv2.imshow("DNI", finalID)
    cv2.imwrite("Imagenes/finalID_back.jpg", finalID)
    # We get all the data from the back part of the ID in different images
    cv2.imshow("MRZ", mrzExtract(finalID))

    cv2.waitKey(0)
    print("Cerrando programa...")
    return 0


if __name__ == "__main__":
    main()