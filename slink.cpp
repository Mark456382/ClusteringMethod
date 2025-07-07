#include <cmath>
#include <vector>
#include <limits>
#include <cstring>

// Экспорт для DLL
extern "C" {

struct Point {
    double x;
    double y;
};

// Расчёт евклидова расстояния между двумя точками
double euclidean_distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Основной алгоритм аггломеративной кластеризации с Single Linkage
__declspec(dllexport)
void slink(Point* points, int n_points, int* cluster_labels, int ) {
    if (n_points <= 0) return;

    // Изначально каждая точка — это отдельный кластер
    for (int i = 0; i < n_points; ++i) {
        cluster_labels[i] = i;
    }

    // Инициализация матрицы расстояний
    std::vector<std::vector<double>> distance(n_points, std::vector<double>(n_points, 0.0));
    for (int i = 0; i < n_points; ++i) {
        for (int j = i + 1; j < n_points; ++j) {
            double dist = euclidean_distance(points[i], points[j]);
            distance[i][j] = dist;
            distance[j][i] = dist;
        }
    }

    // Простая аггломеративная логика с Single Linkage
    int next_cluster_id = 0;
    int current_clusters = n_points;

    // Ограничим количество кластеров, например, 2 — как демонстрация
    while (current_clusters > 2) {
        double min_dist = std::numeric_limits<double>::max();
        int merge_a = -1, merge_b = -1;

        // Поиск двух ближайших кластеров
        for (int i = 0; i < n_points; ++i) {
            for (int j = i + 1; j < n_points; ++j) {
                if (cluster_labels[i] != cluster_labels[j] && distance[i][j] < min_dist) {
                    min_dist = distance[i][j];
                    merge_a = cluster_labels[i];
                    merge_b = cluster_labels[j];
                }
            }
        }

        if (merge_a == -1 || merge_b == -1) break;

        // Объединение кластеров: переименовать все точки с label == merge_b → merge_a
        for (int i = 0; i < n_points; ++i) {
            if (cluster_labels[i] == merge_b) {
                cluster_labels[i] = merge_a;
            }
        }

        current_clusters--;
    }

    // Пронумеровать кластеры заново от 0
    std::vector<int> old_to_new(10000, -1);
    int new_id = 0;
    for (int i = 0; i < n_points; ++i) {
        int lbl = cluster_labels[i];
        if (old_to_new[lbl] == -1) {
            old_to_new[lbl] = new_id++;
        }
        cluster_labels[i] = old_to_new[lbl];
    }
}

} // extern "C"
