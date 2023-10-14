from Interfaz_Code import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
from ultralytics import YOLO

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        # Inicializa las clases predeterminadas
        self.classes_default = 0

        # Conecta los eventos a los métodos
        self.Encendido.clicked.connect(self.start_video)
        self.Apagado.clicked.connect(self.cancel)
        self.Exit.clicked.connect(self.salir)

        self.CheckVest.stateChanged.connect(self.ClassVest)
        self.CheckHat.stateChanged.connect(self.ClassHat)
        self.ChekMask.stateChanged.connect(self.ClassMask)

        # Crea una instancia de Work y pasa la instancia actual de MainWindow
        self.Work = Work(self)

        self.Work.thread_stopped.connect(self.thread_stopped_handler)

    def start_video(self):
        if not self.Work.isRunning():
            self.Work.start()
            self.Work.Imageupd.connect(self.Imageupd_slot)

    def Imageupd_slot(self, Image):
        self.CuadroCam.setPixmap(QPixmap.fromImage(Image))

    def cancel(self):
        if self.Work.isRunning():  # Verifica si el hilo está en funcionamiento
            self.Work.stop()
            self.Work.wait()  # Espera a que el hilo se detenga por completo
        self.CuadroCam.clear()

    def salir(self):
        sys.exit()

    def ClassVest(self, checked):
        if checked:
            self.Work.add_selected_classes([4, 5, 7])
        else:
            self.Work.remove_selected_classes([4, 5, 7])

    def ClassHat(self, checked):
        if checked:
            self.Work.add_selected_classes([0, 2, 5])
        else:
            self.Work.remove_selected_classes([0, 2, 5])

    def ClassMask(self, checked):
        if checked:
            self.Work.add_selected_classes([1, 3, 5])
        else:
            self.Work.remove_selected_classes([1, 3, 5])

    def thread_stopped_handler(self):
        print("Hilo detenido")   

class Work(QThread):
    Imageupd = pyqtSignal(QImage)
    thread_stopped = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.hilo_corriendo = True
        self.main_window = main_window
        self.selected_classes = [5]

    def run(self):
        self.hilo_corriendo = True
        model = YOLO("best.pt")
        cap = cv2.VideoCapture(0)
        while self.hilo_corriendo:
            ret, frame = cap.read()
            if ret:
                resultados = model.predict(frame, imgsz=640, conf=0.60, classes=self.selected_classes)
                anotaciones = resultados[0].plot()
                Image = cv2.cvtColor(anotaciones, cv2.COLOR_BGR2RGB)
                if self.hilo_corriendo:
                    convertir_QT = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                    pic = convertir_QT.scaled(480, 340, Qt.KeepAspectRatio)
                    self.Imageupd.emit(pic)
                NumDetect = resultados[0].__len__()
                NameDetect = resultados[0].verbose()
        cap.release()
        self.thread_stopped.emit()

    def add_selected_classes(self, classes):
        self.selected_classes.extend(classes)

    def remove_selected_classes(self, classes):
        for class_item in classes:
            if class_item in self.selected_classes:
                self.selected_classes.remove(class_item)

    def stop(self):
        self.hilo_corriendo = False
        self.quit()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())