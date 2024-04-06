import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from mw import Ui_MainWindow
import numpy as np
import random
import prueba
import time

class AdalineThread(QThread):
    update_signal = pyqtSignal(float, float, float,bool,list,str,float,float)

    def __init__(self, parent=None):
        super(AdalineThread, self).__init__(parent)
        self.coordenadas = []
        self.salidas_deseadas = []
        self.limite_de_epocas = 0
        self.factor_de_aprendizaje = 0
        self.w1 = 0
        self.w2 = 0
        self.bias = 0
        self.bandera = True

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
    def run(self):
        #error cuadratico medio
        ecm = 1
        error_objetivo = 0.001
        #condicion de paro 
        while error_objetivo <= ecm and self.limite_de_epocas > 0:
            x1 = 0
            x2 = 0
            y_true = []  # Lista para almacenar las salidas verdaderas
            y_pred = []  # Lista para almacenar las salidas 
            errors = []
            
            #algorimo de entrenamiento
            for i in range(len(self.coordenadas)):
                x1 = self.coordenadas[i][0]
                x2 = self.coordenadas[i][1]
                y = self.w1 * x1 + x2 * self.w2 + self.bias
                #funcion de activacion sigmoide
                y = self.sigmoid(y)
                e = self.salidas_deseadas[i] - y
                y_true.append(self.salidas_deseadas[i])
                y_pred.append(y)

                #actualizacion de los pesos
                self.w1 = round((self.w1 + self.factor_de_aprendizaje * e  * x1), 7)
                self.w2 = round((self.w2 + self.factor_de_aprendizaje * e  * x2), 7)
                self.bias = round((self.bias + self.factor_de_aprendizaje * e), 7)
                errors.append(e)

                        
            pesos = np.array(self.w1)
            pesos = np.append(pesos,self.w2)
            nombre = prueba.plot_decision_boundary(self.coordenadas,self.salidas_deseadas,pesos,self.bias)
            ecm = np.mean(np.array(errors) ** 2)
            # print('error',ecm)
            # print('lista de errrores: ',errors)
            # print('salida',y_pred)
            # print('deseada',y_true)
            # print('epocas',self.limite_de_epocas)

            #mandamos la seña al hilo principal
            self.update_signal.emit(self.w1, self.w2, self.bias,self.bandera,y_pred,nombre,ecm,self.limite_de_epocas)
            self.limite_de_epocas -= 1
            
            time.sleep(2)

        self.bandera = False
        self.update_signal.emit(self.w1, self.w2, self.bias,self.bandera,y_pred,nombre,ecm,self.limite_de_epocas)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filename = None
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 750, 750)
        self.coordenadas = []
        self.salidas_deseadas = []
        self.y_pred = None
        self.limite_de_epocas = 0
        self.factor_de_aprendizaje = 0
        self.w1 = round(random.uniform(0, 1), 7)
        self.w2 = round(random.uniform(0, 1), 7)
        self.bias = round(random.uniform(0, 1), 7)
        self.bandera = True
        self.pixmap_item = None
        self.Cartesiano("plano_cartesiano.png")
        self.ui.pushButton_graficar.clicked.connect(self.grafica)
        self.ui.pushButton_reset.clicked.connect(self.reset)
        self.ui.pushButton_exportar.clicked.connect(self.AbrirArchivo)
        self.ui.pushButton.clicked.connect(self.Archivo_Salidas)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_next_image)

    #seleccion de las entradas
    def AbrirArchivo(self):
        archivo, _ = QFileDialog.getOpenFileName(None, "Seleccionar archivo", "", "Archivos de texto (*.txt)")
        try:
            with open(archivo, 'r') as f:
                for linea in f:
                    x, y = map(float, linea.strip().split(','))
                    # Ajuste de coordenadas
                    self.coordenadas.append((x, y))
            nombre = prueba.plano_cartesiano(self.coordenadas)        
            self.Cartesiano(nombre)

        except Exception as e:
            QMessageBox.warning(self, 'Error', 'Archivo no válido')

    #seleccion de las salidas
    def Archivo_Salidas(self):
        archivo, _ = QFileDialog.getOpenFileName(None, "Seleccionar archivo", "", "Archivos de texto (*.txt)")
        try:
            with open(archivo, 'r') as f:
                for linea in f:
                    linea = linea.strip()
                    self.salidas_deseadas.append(float(linea))

        except Exception as e:
            QMessageBox.warning(self, 'Error', 'Archivo no válido')

    #al trabajar con una interfaz grafica de qt requiere un hilo secundario para actualizar los datos de la interfaz
    #un hilo se encarga de las actualziaciones mientras otro de las operaciones
    def adaline(self):
        self.thread = AdalineThread()
        self.thread.coordenadas = self.coordenadas
        self.thread.salidas_deseadas = self.salidas_deseadas
        self.thread.limite_de_epocas = self.limite_de_epocas
        self.thread.factor_de_aprendizaje = self.factor_de_aprendizaje
        self.thread.bias = self.bias
        self.thread.w1 = self.w1
        self.thread.w2 = self.w2
        self.thread.update_signal.connect(self.actualizar_interfaz)
        self.thread.start()
        print('hola')
        self.timer.start(1500) 


    def actualizar_interfaz(self, w1, w2, bias,bandera,y_pred,filename,error,epocas):
        self.w1 = w1
        self.w2 = w2
        self.bias =bias
        self.ui.lineEdit_w1.setText(str(w1))
        self.ui.lineEdit_w2.setText(str(w2))
        self.ui.lineEdit_bias.setText(str(bias))
        self.bandera = bandera
        self.y_pred = y_pred
        self.filename = filename
        self.ui.lineEdit_error.setText(str(error))
        self.ui.lineEdit_restantes.setText(str(epocas))

    def reset(self):
        self.scene.clear()
        self.coordenadas.clear()
        self.Cartesiano("plano_cartesiano.png")
        self.salidas_deseadas.clear()
        self.w1 = round(random.uniform(0, 1), 5)
        self.w2 = round(random.uniform(0, 1), 5)
        self.bias = round(random.uniform(0, 1), 5)
        self.ui.lineEdit_bias.setText(str(self.bias))
        self.ui.lineEdit_w1.setText(str(self.w1))
        self.ui.lineEdit_w2.setText(str(self.w2))
        self.ui.lineEdit_limite.clear()
        self.ui.lineEdit_restantes.clear()
        self.ui.lineEdit_error.clear()


    def show_next_image(self):
        if self.bandera:
            self.filename
            self.Cartesiano(self.filename)
            self.contador += 1
        else:
            self.timer.stop()
    def grafica(self):
        if self.validacion():
            self.adaline()
            
    def validacion(self):
        try:
            self.factor_de_aprendizaje = float(self.ui.lineEdit_factor.text())
            self.limite_de_epocas = float(self.ui.lineEdit_limite.text())
            if self.limite_de_epocas < 0:
                QMessageBox.warning(self, 'Captura no válida', 'Ingrese solo números enteros o reales positivos.')
                return False
            if len(self.coordenadas) == 0:
                QMessageBox.warning(self, 'Ingrese entradas', 'Seleccione entradas en el plano')
                return False
            return True
        except ValueError:
            QMessageBox.warning(self, 'Captura no válida', 'Ingrese solo números enteros o reales positivos.')
            return False

    def Cartesiano(self, filename):
        pixmap = QPixmap(filename)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.ui.graphicsView.setScene(self.scene)

        self.ui.lineEdit_bias.setText(str(self.bias))
        self.ui.lineEdit_w1.setText(str(self.w1))
        self.ui.lineEdit_w2.setText(str(self.w2))

app = QApplication(sys.argv)
ventana = Window()
ventana.show()
sys.exit(app.exec_())
