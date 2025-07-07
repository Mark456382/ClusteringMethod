import ctypes
import numpy as np

# Загрузка DLL
slink = ctypes.CDLL(r'./slink.dll')

# Структура Point
class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

# Обёртка
def run_slink(points_np: np.ndarray):
    n = len(points_np)
    point_array = (Point * n)(*[
        Point(float(p[0]), float(p[1])) for p in points_np
    ])
    labels = (ctypes.c_int * n)()

    slink.slink(point_array, ctypes.c_int(n), labels)

    return list(labels)
