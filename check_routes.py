# oculuz/src/data/dataset/visualize_routes.py
import math
import os
# Установка переменной окружения для Qt5Agg должна быть до импорта matplotlib.pyplot
# Проверьте, действительно ли это необходимо для вашей среды.
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'.venv/Lib/site-packages/PyQt5/Qt5/plugins' # Закомментировано, т.к. специфично для venv
import matplotlib

try:
    matplotlib.use('Qt5Agg')
    print(f"Using Matplotlib backend: {matplotlib.get_backend()}")
except ImportError:
    print("Qt5Agg backend not available. Using default Matplotlib backend.")

import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any

import sys
# Добавляем корневую папку проекта в sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Используем новый ConfigLoader
from configuration.config_loader import ConfigLoader
from src.data.dataset.route_generators import (
    DirectRouteGenerator, CircleRouteGenerator, ArcRouteGenerator, RandomWalkRouteGenerator
)
from src.utils.geometry import EARTH_RADIUS_METERS

logger = logging.getLogger(__name__)

def plot_route_and_source(
        points: List[Dict[str, float]],
        source: Dict[str, float],
        title: str,
        ax: plt.Axes,
        common_config_loader: ConfigLoader # Теперь это ConfigLoader
):
    lats = [p['latitude'] for p in points]
    lons = [p['longitude'] for p in points]

    ax.plot(lons, lats, marker='o', linestyle='-', label='Route Points', markersize=5, alpha=0.7)
    if points:
        ax.plot(lons[0], lats[0], marker='^', color='green', markersize=10, label='Start Point')
    ax.plot(source['longitude'], source['latitude'], marker='*', color='red', markersize=15, label='Source')

    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")

    try:
        base_latitude = common_config_loader["area_definition"]["base_latitude"]
        # area_radius_meters = common_config_loader["area_definition"]["area_radius_meters"]
        # angular_radius_deg = (area_radius_meters / EARTH_RADIUS_METERS) * (180 / math.pi)
        # lon_scale_factor = math.cos(math.radians(base_latitude))
        # ax.set_xlim(common_config_loader["area_definition"]["base_longitude"] - angular_radius_deg / lon_scale_factor,
        #             common_config_loader["area_definition"]["base_longitude"] + angular_radius_deg / lon_scale_factor)
        # ax.set_ylim(base_latitude - angular_radius_deg, base_latitude + angular_radius_deg)
    except KeyError as e:
        logger.warning(f"Could not set plot limits due to missing key in common_config: {e}")

    all_lons = lons + [source['longitude']]
    all_lats = lats + [source['latitude']]
    if not all_lons or not all_lats:
        ax.legend()
        ax.grid(True)
        return

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    delta_lon = max_lon - min_lon
    delta_lat = max_lat - min_lat
    padding_lon = delta_lon * 0.1 if delta_lon > 1e-6 else 0.01
    padding_lat = delta_lat * 0.1 if delta_lat > 1e-6 else 0.01

    ax.set_xlim(min_lon - padding_lon, max_lon + padding_lon)
    ax.set_ylim(min_lat - padding_lat, max_lat + padding_lat)

    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True)


def main():
    logger.info("Starting route visualization script...")

    # 1. Загружаем CommonRouteConfig с помощью ConfigLoader
    common_config_path = "configuration/route_generators/common_route_config.yaml"
    try:
        common_cfg_loader = ConfigLoader(common_config_path)
        logger.info(f"CommonRouteConfig loaded from {common_config_path}. "
                    f"Base Latitude: {common_cfg_loader['area_definition']['base_latitude']}")
    except Exception as e:
        logger.error(f"Failed to load common route config: {e}", exc_info=True)
        return

    # 2. Определяем пути к специфичным конфигам
    route_configs_info = {
        "Direct Route": {
            "path": "configuration/route_generators/direct_route_config.yaml",
            "class": DirectRouteGenerator
        },
        "Circle Route": {
            "path": "configuration/route_generators/circle_route_config.yaml",
            "class": CircleRouteGenerator
        },
        "Arc Route": {
            "path": "configuration/route_generators/arc_route_config.yaml",
            "class": ArcRouteGenerator
        },
        "Random Walk Route": {
            "path": "configuration/route_generators/random_walk_route_config.yaml",
            "class": RandomWalkRouteGenerator
        }
    }

    generators_to_visualize = []
    for title, info in route_configs_info.items():
        try:
            specific_cfg_loader = ConfigLoader(info["path"])
            generator_instance = info["class"](
                common_config=common_cfg_loader,  # Передаем загруженный common_cfg_loader
                specific_config=specific_cfg_loader # Передаем загруженный specific_cfg_loader
            )
            generators_to_visualize.append(
                (title, generator_instance, common_cfg_loader) # Передаем common_cfg_loader для plot_route_and_source
            )
            logger.info(f"Successfully initialized generator: {title}")
        except Exception as e:
            logger.error(f"Failed to initialize generator {title} with config {info['path']}: {e}", exc_info=True)
            continue # Пропускаем этот генератор, если не удалось инициализировать

    if not generators_to_visualize:
        logger.error("No generators were successfully initialized. Exiting.")
        return

    num_generators = len(generators_to_visualize)
    ncols = 2
    nrows = (num_generators + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 7 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, (title, generator_instance, common_loader_for_plot) in enumerate(generators_to_visualize):
        logger.info(f"Generating and plotting: {title}")
        try:
            path_data, source_data = generator_instance.generate_route()
            if not path_data:
                logger.warning(f"No data generated for {title}. Skipping plot.")
                axes[i].set_title(f"{title} (No data)")
                axes[i].text(0.5, 0.5, "No data generated", ha='center', va='center', fontsize=9)
                axes[i].grid(True)
                continue
            plot_route_and_source(path_data, source_data, title, axes[i], common_loader_for_plot)
        except Exception as e:
            logger.error(f"Error during generation or plotting for {title}: {e}", exc_info=True)
            axes[i].set_title(f"{title} (Error)")
            axes[i].text(0.5, 0.5, "Error in generation/plot", ha='center', va='center', color='red', fontsize=9)
            axes[i].grid(True)


    for j in range(num_generators, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_viz_dir = "oculuz_visualizations"
    os.makedirs(output_viz_dir, exist_ok=True)
    plot_filename = os.path.join(output_viz_dir, "generated_routes_examples.png")

    try:
        plt.savefig(plot_filename)
        logger.info(f"Route visualization saved to {plot_filename}")
        plt.show()
    except Exception as e:
        logger.error(f"Error saving or showing plot: {e}", exc_info=True)

    logger.info("Route visualization finished.")

if __name__ == "__main__":
    # Устанавливаем более детальное логирование для отладки самого скрипта
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
    main()