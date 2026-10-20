"""Constants shared by the simulation, the brain and the test range.

Screen units: a position on screen measured from the crosshair, divided by
half the screen width. The crosshair is (0, 0), the left/right edges are
-1/+1, and y points down. Both axes use the same scale.
"""
import math

DEG_PER_COUNT = 0.022          # degrees turned per mouse count at sensitivity 1
SENS_MIN, SENS_MAX = 0.01, 10.0
ASPECT = 16 / 9

# Mouse gain G = screen units moved per mouse count. Across 0.01-10 sens and
# 60-110 FOV it spans roughly 2e-6 .. 2e-2, so the starting guess sits in the
# middle of that range on a log scale.
LOG_G0 = math.log(2e-4)

N_DELAYS = 5                   # the estimator considers 0..4 frames of delay
HIST = 64                      # frames of mouse history kept

OBS_DIM = 57                   # features the network sees; older models used the first 27 or 49
OBS_DIM_V1 = 27                # the original inputs
ACT_DIM = 5                    # gain fix, aim offset x/y, snap-vs-track, click
