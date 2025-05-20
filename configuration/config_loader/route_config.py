# oculuz/configuration/config_loader/route_config.py
import logging
import math
from typing import Dict, Any, Tuple, Optional
from .base_config import BaseConfig

logger = logging.getLogger(__name__)


class CommonRouteConfig(BaseConfig):
    min_points: int
    max_points: int
    min_step_meters: float
    max_step_meters: float
    base_latitude: float
    base_longitude: float
    area_radius_meters: float
    fspl_k: float

    def _set_defaults(self) -> None:
        self.min_points = 5
        self.max_points = 20
        self.min_step_meters = 5.0
        self.max_step_meters = 50.0
        self.base_latitude = 51.5074
        self.base_longitude = -0.1278
        self.area_radius_meters = 5000.0
        self.fspl_k = -20.05
        logger.debug(f"[{self.__class__.__name__}] Python defaults set: {self._to_dict_safe_common_fields()}")

    def _validate_config(self) -> None:
        cls_name = self.__class__.__name__
        if not (hasattr(self, 'min_points') and isinstance(self.min_points, int) and self.min_points > 0):
            raise ValueError(
                f"[{cls_name}] min_points ({getattr(self, 'min_points', 'N/A')}) должен быть положительным целым числом.")
        if not (hasattr(self, 'max_points') and isinstance(self.max_points,
                                                           int) and self.max_points >= self.min_points):
            raise ValueError(
                f"[{cls_name}] max_points ({getattr(self, 'max_points', 'N/A')}) должен быть >= min_points ({getattr(self, 'min_points', 'N/A')}).")
        if not (hasattr(self, 'min_step_meters') and isinstance(self.min_step_meters,
                                                                (float, int)) and self.min_step_meters > 0):
            raise ValueError(
                f"[{cls_name}] min_step_meters ({getattr(self, 'min_step_meters', 'N/A')}) должен быть положительным числом.")
        if not (hasattr(self, 'max_step_meters') and isinstance(self.max_step_meters, (
        float, int)) and self.max_step_meters >= self.min_step_meters):
            raise ValueError(
                f"[{cls_name}] max_step_meters ({getattr(self, 'max_step_meters', 'N/A')}) должен быть >= min_step_meters ({getattr(self, 'min_step_meters', 'N/A')}).")
        if not (hasattr(self, 'base_latitude') and isinstance(self.base_latitude,
                                                              (float, int)) and -90 <= self.base_latitude <= 90):
            raise ValueError(
                f"[{cls_name}] base_latitude ({getattr(self, 'base_latitude', 'N/A')}) должен быть в диапазоне [-90, 90].")
        if not (hasattr(self, 'base_longitude') and isinstance(self.base_longitude,
                                                               (float, int)) and -180 <= self.base_longitude <= 180):
            raise ValueError(
                f"[{cls_name}] base_longitude ({getattr(self, 'base_longitude', 'N/A')}) должен быть в диапазоне [-180, 180].")
        if not (hasattr(self, 'area_radius_meters') and isinstance(self.area_radius_meters,
                                                                   (float, int)) and self.area_radius_meters > 0):
            raise ValueError(
                f"[{cls_name}] area_radius_meters ({getattr(self, 'area_radius_meters', 'N/A')}) должен быть положительным числом.")
        if not (hasattr(self, 'fspl_k') and isinstance(self.fspl_k, (float, int))):
            raise ValueError(f"[{cls_name}] fspl_k ({getattr(self, 'fspl_k', 'N/A')}) должен быть числом.")
        logger.debug(f"[{cls_name}] Common fields validated successfully: {self._to_dict_safe_common_fields()}")

    def _to_dict_safe_common_fields(self) -> Dict[str, Any]:  # Для логгирования
        return {
            "min_points": getattr(self, 'min_points', 'N/A'), "max_points": getattr(self, 'max_points', 'N/A'),
            "min_step_meters": getattr(self, 'min_step_meters', 'N/A'),
            "max_step_meters": getattr(self, 'max_step_meters', 'N/A'),
            "base_latitude": getattr(self, 'base_latitude', 'N/A'),
            "base_longitude": getattr(self, 'base_longitude', 'N/A'),
            "area_radius_meters": getattr(self, 'area_radius_meters', 'N/A'), "fspl_k": getattr(self, 'fspl_k', 'N/A'),
        }

    def _to_dict(self) -> Dict[str, Any]:
        return self._to_dict_safe_common_fields()

    def _from_dict(self, data: Dict[str, Any]) -> None:
        # `self` уже имеет Python-дефолты на этот момент
        self.min_points = int(data.get("min_points", self.min_points))
        self.max_points = int(data.get("max_points", self.max_points))
        self.min_step_meters = float(data.get("min_step_meters", self.min_step_meters))
        self.max_step_meters = float(data.get("max_step_meters", self.max_step_meters))
        self.base_latitude = float(data.get("base_latitude", self.base_latitude))
        self.base_longitude = float(data.get("base_longitude", self.base_longitude))
        self.area_radius_meters = float(data.get("area_radius_meters", self.area_radius_meters))
        self.fspl_k = float(data.get("fspl_k", self.fspl_k))
        logger.debug(
            f"[{self.__class__.__name__}] Values after _from_dict (from its own YAML): {self._to_dict_safe_common_fields()}")


class DirectRouteConfig(CommonRouteConfig):
    def _set_defaults(self) -> None:
        super()._set_defaults()
        # Нет специфичных Python-дефолтов для DirectRouteConfig
        logger.debug(f"[{self.__class__.__name__}] Python defaults set (inherits all from Common).")

    def _validate_config(self) -> None:
        super()._validate_config()  # Валидируем поля CommonRouteConfig
        # Нет специфичных полей для валидации
        logger.debug(f"[{self.__class__.__name__}] Validated (no specific fields beyond Common).")

    def _to_dict(self) -> Dict[str, Any]:
        return super()._to_dict()  # Все поля из CommonRouteConfig

    def _from_dict(self, data: Dict[str, Any]) -> None:
        # `data` - это данные из YAML-файла DirectRouteConfig (может быть пустым)

        # 1. Устанавливаем базовые значения из актуального CommonRouteConfig синглтона.
        # Эти значения уже должны быть в `self` благодаря наследованию Python-дефолтов,
        # но если CommonRouteConfig был загружен из YAML, мы хотим эти значения.
        common_instance = CommonRouteConfig.get_instance()
        self.min_points = common_instance.min_points
        self.max_points = common_instance.max_points
        self.min_step_meters = common_instance.min_step_meters
        self.max_step_meters = common_instance.max_step_meters
        self.base_latitude = common_instance.base_latitude
        self.base_longitude = common_instance.base_longitude
        self.area_radius_meters = common_instance.area_radius_meters
        self.fspl_k = common_instance.fspl_k
        logger.debug(
            f"[{self.__class__.__name__}] Initialized with current values from CommonRouteConfig singleton: {self._to_dict_safe_common_fields()}")

        # 2. Если `data` (из direct_route_config.yaml) содержит переопределения для common-полей, применяем их.
        if data:  # YAML не пуст
            self.min_points = int(data.get("min_points", self.min_points))
            self.max_points = int(data.get("max_points", self.max_points))
            self.min_step_meters = float(data.get("min_step_meters", self.min_step_meters))
            self.max_step_meters = float(data.get("max_step_meters", self.max_step_meters))
            self.base_latitude = float(data.get("base_latitude", self.base_latitude))
            self.base_longitude = float(data.get("base_longitude", self.base_longitude))
            self.area_radius_meters = float(data.get("area_radius_meters", self.area_radius_meters))
            self.fspl_k = float(data.get("fspl_k", self.fspl_k))
            logger.debug(
                f"[{self.__class__.__name__}] Applied overrides from its own YAML. Current values: {self._to_dict_safe_common_fields()}")
        else:
            logger.debug(
                f"[{self.__class__.__name__}] Its own YAML is empty or no data. Using values from CommonRouteConfig singleton.")


class CircleRouteConfig(CommonRouteConfig):
    radius_range_meters: Tuple[float, float]
    center_offset_ratio_max: float
    point_angle_perturbation_deg: float
    point_radius_perturbation_ratio: float
    source_placement_probabilities: Dict[str, float]

    def _set_defaults(self) -> None:
        super()._set_defaults()
        self.radius_range_meters = (100.0, 500.0)
        self.center_offset_ratio_max = 0.5
        self.point_angle_perturbation_deg = 5.0
        self.point_radius_perturbation_ratio = 0.1
        self.source_placement_probabilities = {"inside": 0.6, "outside": 0.3, "center": 0.1}
        logger.debug(f"[{self.__class__.__name__}] Python defaults set for specific fields.")

    def _validate_config(self) -> None:
        super()._validate_config()
        cls_name = self.__class__.__name__
        if not (hasattr(self, 'radius_range_meters') and isinstance(self.radius_range_meters, tuple) and len(
                self.radius_range_meters) == 2 and
                all(isinstance(x, (float, int)) and x > 0 for x in self.radius_range_meters) and
                self.radius_range_meters[0] <= self.radius_range_meters[1]):
            raise ValueError(
                f"[{cls_name}] radius_range_meters ({getattr(self, 'radius_range_meters', 'N/A')}) некорректен.")
        if not (hasattr(self, 'center_offset_ratio_max') and isinstance(self.center_offset_ratio_max, (
        float, int)) and 0 <= self.center_offset_ratio_max <= 1):
            raise ValueError(
                f"[{cls_name}] center_offset_ratio_max ({getattr(self, 'center_offset_ratio_max', 'N/A')}) должен быть в [0, 1].")
        if not (hasattr(self, 'point_angle_perturbation_deg') and isinstance(self.point_angle_perturbation_deg, (
        float, int)) and self.point_angle_perturbation_deg >= 0):
            raise ValueError(
                f"[{cls_name}] point_angle_perturbation_deg ({getattr(self, 'point_angle_perturbation_deg', 'N/A')}) должен быть >= 0.")
        if not (hasattr(self, 'point_radius_perturbation_ratio') and isinstance(self.point_radius_perturbation_ratio, (
        float, int)) and 0 <= self.point_radius_perturbation_ratio <= 1):
            raise ValueError(
                f"[{cls_name}] point_radius_perturbation_ratio ({getattr(self, 'point_radius_perturbation_ratio', 'N/A')}) должен быть в [0,1].")
        if not (hasattr(self, 'source_placement_probabilities') and
                isinstance(self.source_placement_probabilities, dict) and
                math.isclose(sum(self.source_placement_probabilities.values()), 1.0) and
                all(k in ["inside", "outside", "center"] for k in self.source_placement_probabilities)):
            prob_sum = sum(self.source_placement_probabilities.values()) if hasattr(self,
                                                                                    'source_placement_probabilities') and isinstance(
                self.source_placement_probabilities, dict) else "N/A"
            raise ValueError(
                f"[{cls_name}] source_placement_probabilities ({getattr(self, 'source_placement_probabilities', 'N/A')}) некорректен. Сумма: {prob_sum}")
        logger.debug(f"[{cls_name}] Specific fields validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        common_dict = super()._to_dict()
        common_dict.update({
            "radius_range_meters": list(self.radius_range_meters) if hasattr(self, 'radius_range_meters') else 'N/A',
            "center_offset_ratio_max": getattr(self, 'center_offset_ratio_max', 'N/A'),
            "point_angle_perturbation_deg": getattr(self, 'point_angle_perturbation_deg', 'N/A'),
            "point_radius_perturbation_ratio": getattr(self, 'point_radius_perturbation_ratio', 'N/A'),
            "source_placement_probabilities": getattr(self, 'source_placement_probabilities', 'N/A'),
        })
        return common_dict

    def _from_dict(self, data: Dict[str, Any]) -> None:
        # 1. Инициализируем common поля из CommonRouteConfig синглтона
        common_instance = CommonRouteConfig.get_instance()
        # Присваиваем common поля СЕБЕ из common_instance
        for key, value in common_instance._to_dict().items():
            setattr(self, key, value)
        logger.debug(
            f"[{self.__class__.__name__}] Initialized common fields from CommonRouteConfig singleton: {super()._to_dict_safe_common_fields()}")

        # 2. Устанавливаем/переопределяем СВОИ специфичные поля и common-поля из `data` (YAML текущего класса)
        #    `self` уже имеет Python-дефолты для своих полей + значения из common_instance для общих полей.
        if data:  # YAML не пуст
            # Переопределение common полей, если они есть в YAML этого класса
            self.min_points = int(data.get("min_points", self.min_points))
            self.max_points = int(data.get("max_points", self.max_points))
            self.min_step_meters = float(data.get("min_step_meters", self.min_step_meters))
            self.max_step_meters = float(data.get("max_step_meters", self.max_step_meters))
            self.base_latitude = float(data.get("base_latitude", self.base_latitude))
            self.base_longitude = float(data.get("base_longitude", self.base_longitude))
            self.area_radius_meters = float(data.get("area_radius_meters", self.area_radius_meters))
            self.fspl_k = float(data.get("fspl_k", self.fspl_k))

            # Установка/переопределение СВОИХ полей
            self.radius_range_meters = tuple(data.get("radius_range_meters", list(self.radius_range_meters)))
            self.center_offset_ratio_max = float(data.get("center_offset_ratio_max", self.center_offset_ratio_max))
            self.point_angle_perturbation_deg = float(
                data.get("point_angle_perturbation_deg", self.point_angle_perturbation_deg))
            self.point_radius_perturbation_ratio = float(
                data.get("point_radius_perturbation_ratio", self.point_radius_perturbation_ratio))
            self.source_placement_probabilities = data.get("source_placement_probabilities",
                                                           self.source_placement_probabilities)
            logger.debug(f"[{self.__class__.__name__}] Applied overrides from its own YAML.")
        else:
            logger.debug(
                f"[{self.__class__.__name__}] Its own YAML is empty or no data. Using values from CommonRouteConfig and own Python defaults.")
        logger.debug(f"[{self.__class__.__name__}] State after _from_dict: {self._to_dict()}")


# --- ArcRouteConfig ---
class ArcRouteConfig(CommonRouteConfig):
    parabola_coeff_range: Tuple[float, float]
    arc_length_ratio_range: Tuple[float, float]

    def _set_defaults(self) -> None:
        super()._set_defaults()
        self.parabola_coeff_range = (0.001, 0.01)
        self.arc_length_ratio_range = (0.5, 1.0)
        logger.debug(f"[{self.__class__.__name__}] Python defaults set for specific fields.")

    def _validate_config(self) -> None:
        super()._validate_config()
        cls_name = self.__class__.__name__
        if not (hasattr(self, 'parabola_coeff_range') and isinstance(self.parabola_coeff_range, tuple) and len(
                self.parabola_coeff_range) == 2 and
                all(isinstance(x, (float, int)) and x != 0 for x in self.parabola_coeff_range)):
            raise ValueError(
                f"[{cls_name}] parabola_coeff_range ({getattr(self, 'parabola_coeff_range', 'N/A')}) некорректен.")
        if not (hasattr(self, 'arc_length_ratio_range') and isinstance(self.arc_length_ratio_range, tuple) and len(
                self.arc_length_ratio_range) == 2 and
                all(0 < x <= 1 for x in self.arc_length_ratio_range) and
                self.arc_length_ratio_range[0] <= self.arc_length_ratio_range[1]):
            raise ValueError(
                f"[{cls_name}] arc_length_ratio_range ({getattr(self, 'arc_length_ratio_range', 'N/A')}) некорректен.")
        logger.debug(f"[{cls_name}] Specific fields validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        common_dict = super()._to_dict()
        common_dict.update({
            "parabola_coeff_range": list(self.parabola_coeff_range) if hasattr(self, 'parabola_coeff_range') else 'N/A',
            "arc_length_ratio_range": list(self.arc_length_ratio_range) if hasattr(self,
                                                                                   'arc_length_ratio_range') else 'N/A',
        })
        return common_dict

    def _from_dict(self, data: Dict[str, Any]) -> None:
        common_instance = CommonRouteConfig.get_instance()
        for key, value in common_instance._to_dict().items():
            setattr(self, key, value)
        logger.debug(f"[{self.__class__.__name__}] Initialized common fields from CommonRouteConfig singleton.")

        if data:
            self.min_points = int(data.get("min_points", self.min_points))
            self.max_points = int(data.get("max_points", self.max_points))
            # ... (остальные common поля) ...
            self.min_step_meters = float(data.get("min_step_meters", self.min_step_meters))
            self.max_step_meters = float(data.get("max_step_meters", self.max_step_meters))
            self.base_latitude = float(data.get("base_latitude", self.base_latitude))
            self.base_longitude = float(data.get("base_longitude", self.base_longitude))
            self.area_radius_meters = float(data.get("area_radius_meters", self.area_radius_meters))
            self.fspl_k = float(data.get("fspl_k", self.fspl_k))

            self.parabola_coeff_range = tuple(data.get("parabola_coeff_range", list(self.parabola_coeff_range)))
            self.arc_length_ratio_range = tuple(data.get("arc_length_ratio_range", list(self.arc_length_ratio_range)))
            logger.debug(f"[{self.__class__.__name__}] Applied overrides from its own YAML.")
        else:
            logger.debug(f"[{self.__class__.__name__}] Its own YAML is empty or no data.")
        logger.debug(f"[{self.__class__.__name__}] State after _from_dict: {self._to_dict()}")


# --- RandomWalkRouteConfig ---
class RandomWalkRouteConfig(CommonRouteConfig):
    turn_probability: float
    turn_angle_range_deg: Tuple[float, float]

    def _set_defaults(self) -> None:
        super()._set_defaults()
        self.turn_probability = 0.3
        self.turn_angle_range_deg = (-60.0, 60.0)
        logger.debug(f"[{self.__class__.__name__}] Python defaults set for specific fields.")

    def _validate_config(self) -> None:
        super()._validate_config()
        cls_name = self.__class__.__name__
        if not (hasattr(self, 'turn_probability') and isinstance(self.turn_probability,
                                                                 (float, int)) and 0 <= self.turn_probability <= 1):
            raise ValueError(
                f"[{cls_name}] turn_probability ({getattr(self, 'turn_probability', 'N/A')}) должен быть в [0, 1].")
        if not (hasattr(self, 'turn_angle_range_deg') and isinstance(self.turn_angle_range_deg, tuple) and len(
                self.turn_angle_range_deg) == 2 and
                -360 < self.turn_angle_range_deg[0] <= self.turn_angle_range_deg[
                    1] < 360):  # Убрал < 360 для max, если нужно полный оборот
            raise ValueError(
                f"[{cls_name}] turn_angle_range_deg ({getattr(self, 'turn_angle_range_deg', 'N/A')}) некорректен.")
        logger.debug(f"[{cls_name}] Specific fields validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        common_dict = super()._to_dict()
        common_dict.update({
            "turn_probability": getattr(self, 'turn_probability', 'N/A'),
            "turn_angle_range_deg": list(self.turn_angle_range_deg) if hasattr(self, 'turn_angle_range_deg') else 'N/A',
        })
        return common_dict

    def _from_dict(self, data: Dict[str, Any]) -> None:
        common_instance = CommonRouteConfig.get_instance()
        for key, value in common_instance._to_dict().items():
            setattr(self, key, value)
        logger.debug(f"[{self.__class__.__name__}] Initialized common fields from CommonRouteConfig singleton.")

        if data:
            self.min_points = int(data.get("min_points", self.min_points))
            self.max_points = int(data.get("max_points", self.max_points))
            # ... (остальные common поля) ...
            self.min_step_meters = float(data.get("min_step_meters", self.min_step_meters))
            self.max_step_meters = float(data.get("max_step_meters", self.max_step_meters))
            self.base_latitude = float(data.get("base_latitude", self.base_latitude))
            self.base_longitude = float(data.get("base_longitude", self.base_longitude))
            self.area_radius_meters = float(data.get("area_radius_meters", self.area_radius_meters))
            self.fspl_k = float(data.get("fspl_k", self.fspl_k))

            self.turn_probability = float(data.get("turn_probability", self.turn_probability))
            self.turn_angle_range_deg = tuple(data.get("turn_angle_range_deg", list(self.turn_angle_range_deg)))
            logger.debug(f"[{self.__class__.__name__}] Applied overrides from its own YAML.")
        else:
            logger.debug(f"[{self.__class__.__name__}] Its own YAML is empty or no data.")
        logger.debug(f"[{self.__class__.__name__}] State after _from_dict: {self._to_dict()}")