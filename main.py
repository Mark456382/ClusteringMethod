import os
import sys
import csv
import ctypes
import platform
import subprocess
from ctypes import Structure, c_double, c_int
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


class Point(Structure):
    """Структура для C++"""
    _fields_ = [("x", c_double), ("y", c_double)]

class SLinkGUI(QWidget):
    def __init__(self):
        super().__init__()
        matplotlib.use("Qt5Agg")

        self.setWindowTitle("Single-Linkage Clustering GUI")
        self.setFixedSize(500, 300)  # Фиксированный размер окна

        layout = QVBoxLayout()
        self.label = QLabel("Загрузите CSV-файл с точками")
        layout.addWidget(self.label)

        self.load_button = QPushButton("Загрузить CSV и построить дендрограмму")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        self.scatter_button = QPushButton("Показать точки")
        self.scatter_button.clicked.connect(self.show_scatter)
        layout.addWidget(self.scatter_button)

        self.setLayout(layout)

        self.points = []  # Для хранения загруженных точек
        self.labels = []  # Кластерные метки

        self.dll = ctypes.CDLL("./slink.dll")
        self.dll.slink.argtypes = [ctypes.POINTER(Point), c_int, ctypes.POINTER(c_int)]

    def load_csv(self):
        """Функция загрузки .csv файла"""
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv)")
       
        # Чтение CSV
        points = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    try:
                        x, y = float(row[0]), float(row[1])
                        points.append((x, y))
                    except ValueError:
                        continue

        self.points = points
        n = len(points)
        if n == 0:
            self.label.setText("Файл не содержит точек.")
            log("Файл не содержит точек")
            return

        c_points = (Point * n)(*[
            Point(x, y) for x, y in points
        ])
        labels = (c_int * n)()
        self.dll.slink(c_points, n, labels)
        self.labels = list(labels)

        # Построение дендрограммы
        dist_matrix = self._calculate_distance_matrix(points)
        Z = linkage(dist_matrix, method='single')
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=[f"p{i+1}" for i in range(n)])
        plt.title("Дендрограмма (Single Linkage)")
        plt.xlabel("Точки")
        plt.ylabel("Расстояние")
        plt.tight_layout()
        plt.show()

    def _calculate_distance_matrix(self, points):
        n = len(points)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                dists.append(dist)
        return dists

    def show_scatter(self):
        if not self.points or not self.labels:
            self.label.setText("Сначала загрузите файл и выполните кластеризацию.")
            return
        
        points = np.array(self.points)
        labels = np.array(self.labels)
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        plt.figure(figsize=(8, 6))
        for i, ul in enumerate(unique_labels):
            cluster_points = points[labels == ul]
            for j, (x, y) in enumerate(cluster_points):
                idx = np.where((points[:, 0] == x) & (points[:, 1] == y))[0][0]
                plt.text(x + 0.02, y + 0.02, f"p{idx+1}", fontsize=9)
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        label=f"Кластер {ul}", color=colors[i])

        plt.title("Диаграмма рассеяния точек с подписями")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SLinkGUI()
    window.show()
    sys.exit(app.exec_())
