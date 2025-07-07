import os
import sys
import csv
import msvcrt
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
import matplotlib
matplotlib.use("Qt5Agg")  
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

from slink_interface import run_slink  


@contextmanager
def redirect_stdout_to_file(filename):
    log_file = open(filename, "a", buffering=1)
    fd = sys.stdout.fileno()
    sys.stdout.flush()
    saved_fd = os.dup(fd)
    os.dup2(log_file.fileno(), fd)

    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, fd)
        os.close(saved_fd)
        log_file.close()

class SLinkGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Single-Linkage Clustering GUI")
        self.setFixedSize(500, 300)

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

        self.points = []
        self.labels = []

        if not os.path.exists("result"):
            os.makedirs("result")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv)")
        if not path:
            return

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
            return

        points_np = np.array(points)

        log_path = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        if not os.path.exists("logs"):
            os.makedirs("logs")

        with redirect_stdout_to_file(log_path):
            self.labels = run_slink(points_np)

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
