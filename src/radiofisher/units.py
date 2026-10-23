"""Units and numerical conventions shared by the CHIME forecasts.

The forecast and its tabulated baseline density use c = 300,000 km/s.
Keep this convention together: substituting the exact SI value (299,792.458
km/s) in only one calculation would make the distances and baselines disagree.
"""

from math import pi

FORECAST_LIGHT_SPEED_KM_S = 300_000.0
DEG_TO_RAD = pi / 180.0
HOUR_IN_MHZ_INVERSE = 3600 * 1_000_000  # exact: seconds/hour times Hz/MHz
NU21_MHZ = 1420.406  # 21 cm rest frequency used by the CHIME Overview
