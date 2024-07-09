import lir
import numpy as np
import matplotlib.pyplot as plt
from helper_functions.plotting import tippett

# Code for building the toy example described in chapter 8.2 of the corresponding thesis
validation_lr = np.array([100,5,0.1,1,0.01,10])
validation_truth = np.array([1,1,0,0,0,0])

cllr = lir.metrics.cllr(validation_lr, validation_truth)
cllr_min = lir.metrics.cllr_min(validation_lr, validation_truth)
cllr_cal = cllr - cllr_min
print(f"Toy example results: Cllr: {cllr:.3f}, Cllr_min: {cllr_min:.3f}, Cllr_cal: {cllr_cal:.3f}")

tippett(validation_lr, validation_truth)
plt.show()

with lir.plotting.show() as ax:
    ax.pav(validation_lr, validation_truth)
plt.show()

with lir.plotting.show() as ax:
    ax.ece(validation_lr, validation_truth)
plt.show()

print('')
# Example effect union of sets on Cllrs as described in chapter  8.5 of the corresponding thesis
V1_LR = np.array([10,1])
V1_truth = np.array([1,0])
V2_LR = np.array([1,0.1])
V2_truth = np.array([1,0])
V_LR = np.array([10,1,1,0.1])
V_truth = np.array([1,0,1,0])

cllr = lir.metrics.cllr(V1_LR, V1_truth)
cllr_min = lir.metrics.cllr_min(V1_LR, V1_truth)
cllr_cal = cllr - cllr_min
print(f"V1 results: Cllr: {cllr:.3f}, Cllr_min: {cllr_min:.3f}, Cllr_cal: {cllr_cal:.3f}")

cllr = lir.metrics.cllr(V2_LR, V2_truth)
cllr_min = lir.metrics.cllr_min(V2_LR, V2_truth)
cllr_cal = cllr - cllr_min
print(f"V2 results: Cllr: {cllr:.3f}, Cllr_min: {cllr_min:.3f}, Cllr_cal: {cllr_cal:.3f}")

cllr = lir.metrics.cllr(V_LR, V_truth)
cllr_min = lir.metrics.cllr_min(V_LR, V_truth)
cllr_cal = cllr - cllr_min
print(f"V results: Cllr: {cllr:.3f}, Cllr_min: {cllr_min:.3f}, Cllr_cal: {cllr_cal:.3f}")
