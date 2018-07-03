import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal
import datetime
class ShowVideo(QtCore.QObject):
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer =cv2.face.LBPHFaceRecognizer_create()
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    def writedata(self):
            camera_port = 0
            camera = cv2.VideoCapture(camera_port)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face_recognizer =cv2.face.LBPHFaceRecognizer_create()
            nb = 20
            while (True): 
                ret, img = camera.read()
                only_face = np.array(10)        
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)            
                    only_face = gray[y:y+h,x:x+w]            
                    cv2.imwrite("data/user"+str(nb)+".jpg", only_face) 
                nb = nb + 1         
                cv2.imshow('live video',img)       
                cv2.waitKey(1)        
                if nb == 40:            
                    camera.release()          
                    cv2.destroyAllWindows()            
                    break
                color_swapped_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, _ = color_swapped_image.shape
                qt_image = QtGui.QImage(color_swapped_image.data,width,height,color_swapped_image.strides[0],QtGui.QImage.Format_RGB888)
                self.VideoSignal.emit(qt_image)
    def train_data():
        face_recognizer =cv2.face.LBPHFaceRecognizer_create()
        images = []  
        labels =[]
        for i in range(39):     
            image_pil = Image.open('data/user{}.jpg'.format(i+1)).convert('L')      
            img = np.array(image_pil, 'uint8')    
            faces = face_cascade.detectMultiScale(img)    
            for (x, y, w, h) in faces:    
                images.append(image[y: y + h, x: x + w]) 
                if i<20:       
                    labels.append(1)      
                else:
                    labels.append(2)  
                cv2.imshow("Adding faces to traning set...", img[y: y + h, x: x + w])    
                cv2.waitKey(10)   
        face_recognizer.train(images, np.array(labels))  
        f= open("trainer.yml","w+")
        face_recognizer.save('trainer/trainer.yml') 
        cv2.destroyAllWindows()                                

    def recon_data(self): 
        camera_port = 0
        camera = cv2.VideoCapture(camera_port)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_recognizer =cv2.face.LBPHFaceRecognizer_create()
        print("ok")
        face_recognizer.read('trainer/trainer.yml')
        print("i")
        while True:       
            ret, img =camera.read()      
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
            faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), 
            flags=cv2.CASCADE_SCALE_IMAGE)
            face_recognizer =cv2.face.LBPHFaceRecognizer_create()

            for(x, y, w, h) in faces:          
                cv2.rectangle(img,(x-10,y-10),(x+w+10,y+h+10), (225,255,255),2) 
                id_user, conf = face_recognizer.predict(gray[y:y+h,x:x+w])       
            
                if id_user == 1:              
                    name = "alayasssssssssssss"          
                if id_user == 2:              
                    name = "Tasnim"
                else :
                    name = "indefined"
                cv2.putText(img,str(name), (x,y-15), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, 25)           
                cv2.imshow('im',img)       
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
class ImageViewer(QtWidgets.QWidget):
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.img)
    def setImage(self, img):
        if img.isNull():
            print("Viewer Dropped frame!")
        self.img = img
        if img.size() != self.size():
            self.setFixedSize(img.size())
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    thread = QtCore.QThread()
    thread.start()
    _translate = QtCore.QCoreApplication.translate


    vid = ShowVideo()
    vid.moveToThread(thread)
    image_viewer = ImageViewer()
    vid.VideoSignal.connect(image_viewer.setImage)
    layout_widget = QtWidgets.QWidget()
    layout_widget.setWindowTitle("FALCON TUNISIA")

    #start
    push_button1 =QtWidgets.QPushButton('Start')
    push_button1.clicked.connect(vid.train_data)
    push_button1.clicked.connect(vid.recon_data)
    
    #calendr
    calendarWidget = QtWidgets.QCalendarWidget()
    calendarWidget.setGeometry(QtCore.QRect(180, 50, 411, 171))
    calendarWidget.setStyleSheet("alternate-background-color: rgb(204, 204, 204);\n"
        "background-color: rgb(177, 177, 177);")
    calendarWidget.setObjectName("calendarWidget")
    #symbole
    labelImg = QtWidgets.QLabel()
    labelImg.setPixmap(QtGui.QPixmap("test.png"))
    labelImg.setObjectName("labelImg")
    
    vertical_layout = QtWidgets.QVBoxLayout()
    layout_widget.setLayout(vertical_layout)
    vertical_layout.addWidget(labelImg)
    vertical_layout.addWidget(image_viewer)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(calendarWidget)
    
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())
    layout_widget.show()
