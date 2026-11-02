"""Units and numerical conventions shared by the CHIME forecasts.

The forecast and its tabulated baseline density both use c = 300000 km/s. If
the SI value of 299792.458 km/s is used in only one of them, the distances and
the baselines no longer agree.
"""

from math import pi

FORECAST_LIGHT_SPEED_KM_S = 300_000.0
DEG_TO_RAD = pi / 180.0
HOUR_IN_MHZ_INVERSE = 3600 * 1_000_000  # exact, seconds per hour times Hz per MHz
NU21_MHZ = 1420.406  # 21 cm rest frequency used by the CHIME Overview
