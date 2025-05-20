# oculuz/src/data/dataset/visualize_routes.py
import math

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'.venv/Lib/site-packages/PyQt5/Qt5/plugins'
import matplotlib

matplotlib.use('Qt5Agg')  # Используем Qt5Agg для интерактивности
print(f"Using Matplotlib backend: {matplotlib.get_backend()}")

import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any

# Настройка логирования для скрипта, если он запускается отдельно
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Импорты из проекта oculuz
# Убедитесь, что PYTHONPATH настроен правильно, если запускаете из другой директории
# или добавьте oculuz в sys.path
import sys

# Добавляем корневую папку проекта в sys.path, если запускаем скрипт напрямую
# Это нужно, чтобы работали импорты вида from configuration...
# Предполагается, что скрипт находится в oculuz/src/data/dataset/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configuration.config_loader import (
    CommonRouteConfig, DirectRouteConfig, CircleRouteConfig, ArcRouteConfig, RandomWalkRouteConfig
)
from src.data.dataset.route_generators import (
    DirectRouteGenerator, CircleRouteGenerator, ArcRouteGenerator, RandomWalkRouteGenerator
)
from src.utils.geometry import EARTH_RADIUS_METERS  # для преобразования градусов в метры для масштаба

logger = logging.getLogger(__name__)


def plot_route_and_source(
        points: List[Dict[str, float]],  # [{'latitude': ..., 'longitude': ...}, ...]
        source: Dict[str, float],  # {'latitude': ..., 'longitude': ...}
        title: str,
        ax: plt.Axes,
        common_config_for_area  # CommonRouteConfig для получения base_lat/lon и area_radius
):
    """Отрисовывает маршрут и источник на заданных осях."""
    lats = [p['latitude'] for p in points]
    lons = [p['longitude'] for p in points]

    ax.plot(lons, lats, marker='o', linestyle='-', label='Route Points', markersize=5, alpha=0.7)
    if points:  # Отмечаем начальную точку
        ax.plot(lons[0], lats[0], marker='^', color='green', markersize=10, label='Start Point')
    ax.plot(source['longitude'], source['latitude'], marker='*', color='red', markersize=15, label='Source')

    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")

    # Установка пределов для осей на основе рабочей области
    # Переводим area_radius_meters в градусы (приблизительно)
    # 1 градус широты ~ 111 км. 1 градус долготы ~ 111 км * cos(широты)
    # Для простоты возьмем примерный масштаб
    deg_per_meter_lat = 1 / (EARTH_RADIUS_METERS * (math.pi / 180))  # Неправильно.
    # 1 градус = pi/180 радиан. Длина дуги = R * угол_в_радианах.
    # Угловое расстояние = расстояние_в_метрах / R (в радианах)
    # Угловое расстояние в градусах = (расстояние_в_метрах / R) * (180/pi)

    angular_radius_deg = (common_config_for_area.area_radius_meters / EARTH_RADIUS_METERS) * (180 / math.pi)

    # Коэффициент для долготы из-за схождения меридианов
    lon_scale_factor = math.cos(math.radians(common_config_for_area.base_latitude))

    # ax.set_xlim(common_config_for_area.base_longitude - angular_radius_deg / lon_scale_factor,
    #             common_config_for_area.base_longitude + angular_radius_deg / lon_scale_factor)
    # ax.set_ylim(common_config_for_area.base_latitude - angular_radius_deg,
    #             common_config_for_area.base_latitude + angular_radius_deg)

    # Вместо жестких лимитов, дадим matplotlib немного свободы, но обеспечим квадратность
    # Это важно для визуального восприятия геометрии
    all_lons = lons + [source['longitude']]
    all_lats = lats + [source['latitude']]
    if not all_lons or not all_lats:  # Если точек нет
        ax.legend()
        return

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    delta_lon = max_lon - min_lon
    delta_lat = max_lat - min_lat

    # Добавим отступы
    padding_lon = delta_lon * 0.1 if delta_lon > 1e-6 else 0.01
    padding_lat = delta_lat * 0.1 if delta_lat > 1e-6 else 0.01

    ax.set_xlim(min_lon - padding_lon, max_lon + padding_lon)
    ax.set_ylim(min_lat - padding_lat, max_lat + padding_lat)

    ax.set_aspect('equal', adjustable='box')  # Делает масштаб по осям одинаковым
    ax.legend()
    ax.grid(True)


def main():
    logger.info("Starting route visualization script...")

    # 1. СНАЧАЛА загружаем CommonRouteConfig
    common_config_path = "configuration/route_generators/common_route_config.yaml"
    CommonRouteConfig.load(common_config_path)
    # Логгируем значения из CommonRouteConfig ПОСЛЕ его загрузки
    common_instance_check = CommonRouteConfig.get_instance()
    logger.info(f"CommonRouteConfig loaded. "
                f"min_points: {common_instance_check.min_points}, "
                f"max_points: {common_instance_check.max_points}, "
                f"min_step: {common_instance_check.min_step_meters}, "
                f"max_step: {common_instance_check.max_step_meters}, "
                f"base_lat: {common_instance_check.base_latitude}")

    # 2. Затем загружаем специфичные конфиги.
    # Они при загрузке будут использовать уже обновленный CommonRouteConfig для
    # значений по умолчанию, если специфичный YAML не переопределяет их.
    direct_config_path = "configuration/route_generators/direct_route_config.yaml"
    circle_config_path = "configuration/route_generators/circle_route_config.yaml"
    arc_config_path = "configuration/route_generators/arc_route_config.yaml"
    random_walk_config_path = "configuration/route_generators/random_walk_route_config.yaml"

    # Вызов .load() вернет обновленный синглтон
    direct_cfg_instance = DirectRouteConfig.load(direct_config_path)
    circle_cfg_instance = CircleRouteConfig.load(circle_config_path)
    arc_cfg_instance = ArcRouteConfig.load(arc_config_path)
    random_walk_cfg_instance = RandomWalkRouteConfig.load(random_walk_config_path)

    logger.debug(f"DirectRouteConfig after load: min_points={direct_cfg_instance.min_points}")
    logger.debug(
        f"CircleRouteConfig after load: min_points={circle_cfg_instance.min_points}, radius_range={circle_cfg_instance.radius_range_meters}")

    generators_to_visualize = [
        ("Direct Route", DirectRouteGenerator(config=direct_cfg_instance), direct_cfg_instance),
        ("Circle Route", CircleRouteGenerator(config=circle_cfg_instance), circle_cfg_instance),
        ("Arc Route", ArcRouteGenerator(config=arc_cfg_instance), arc_cfg_instance),
        ("Random Walk Route", RandomWalkRouteGenerator(config=random_walk_cfg_instance), random_walk_cfg_instance)
    ]

    num_generators = len(generators_to_visualize)
    # Рассчитываем nrows более аккуратно
    ncols = 2
    nrows = (num_generators + ncols - 1) // ncols  # Округление вверх
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 7 * nrows),
                             squeeze=False)  # squeeze=False чтобы axes всегда был 2D
    axes = axes.flatten()

    for i, (title, generator_instance, route_conf_instance) in enumerate(generators_to_visualize):
        logger.info(f"Generating and plotting: {title} using config: {route_conf_instance._to_dict()}")
        path_data, source_data = generator_instance.generate_route()

        if not path_data:
            logger.warning(f"No data generated for {title}. Skipping plot.")
            axes[i].set_title(f"{title} (No data)")
            axes[i].text(0.5, 0.5, "No data generated", ha='center', va='center')
            axes[i].grid(True)
            continue

        plot_route_and_source(path_data, source_data, title, axes[i], route_conf_instance)

    # Удаляем лишние subplots, если они есть
    for j in range(num_generators, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()

    output_viz_dir = "oculuz_visualizations"
    os.makedirs(output_viz_dir, exist_ok=True)
    plot_filename = os.path.join(output_viz_dir, "generated_routes_examples.png")
    plt.savefig(plot_filename)
    logger.info(f"Route visualization saved to {plot_filename}")

    plt.show()
    logger.info("Route visualization finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
    main()
