import lir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from plotting import tippett

#Code for building a toy example
'''
def tippett(lrs, y, plot_type=1, ax=plt):
    """
    plots empirical cumulative distribution functions of same-source and
        different-sources lrs

    Parameters
    ----------
    lrs : the likelihood ratios
    y : a numpy array of labels (0 or 1)
    plot_type : an integer, must be either 1 or 2.
        In type 1 both curves show proportion of lrs greater than or equal to the
        x-axis value, while in type 2 the curve for same-source shows the
        proportion of lrs smaller than or equal to the x-axis value.
    ax: axes to plot figure to
    """
    log_lrs = np.log10(lrs)

    lr_0, lr_1 = lir.util.Xy_to_Xn(log_lrs, y)
    xplot = np.linspace(np.min(log_lrs), np.max(log_lrs)+0.0001, 10000)
    perc0 = (sum(i >= xplot for i in lr_0) / len(lr_0)) * 100
    perc1 = (sum(i >= xplot for i in lr_1) / len(lr_1)) * 100


    ax.plot(xplot, perc1, color='b', label='LRs given $\mathregular{H_1}$')
    ax.plot(xplot, perc0, color='r', label='LRs given $\mathregular{H_2}$')
    ax.axvline(x=0, color='k', linestyle='--')
    ax.set_xlabel('log$_{10}$(LR)')
    ax.set_ylabel('Cumulative proportion')
    ax.legend()

n= 99
x = np.linspace(2,10000, num = n)

y = np.random.randn(n)*3 + x**(1/2)
ir = IsotonicRegression(out_of_bounds="clip")
y_hat = ir.fit_transform(x,y)

plt.scatter(x,y)
plt.plot(x,y_hat, color = 'tab:orange')
plt.show()
'''
validation_lr = np.array([100,5,0.1,1,0.01,10])
validation_truth = np.array([1,1,0,0,0,0])
validation_lr = np.array([10,1,1,0.1])
validation_truth = np.array([1,0,1,0])

h1_lrs = validation_lr[validation_truth == 1]
h2_lrs = validation_lr[validation_truth == 0]
cllr = lir.metrics.cllr(validation_lr, validation_truth)
cllr_min = lir.metrics.cllr_min(validation_lr, validation_truth)
cllr_cal = cllr - cllr_min
print(f"Cllr: {cllr:.3f}, Cllr_min: {cllr_min:.3f}, Cllr_cal: {cllr_cal:.3f}")

tippett(validation_lr, validation_truth)
plt.show()

with lir.plotting.show() as ax:
    ax.pav(validation_lr, validation_truth)
plt.show()

with lir.plotting.show() as ax:
    ax.ece(validation_lr, validation_truth)
plt.show()
