"""

Physics 111B: OTZ

python version of FitlogcurvevarX.m,  FitlogcurvevarY.m, and Equipartition.m

PSD Data is 3D arrays of the format [ frequency, X data, Y data ]
FitlogcurvevarX and FitlogcurvevarY are the same, except X reads the 1st column and Y reads the 2nd column

Spring 2026
Julian Vale & Carina Lee

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
# from scipy.signal import welch
# %matplotlib inline

# get PSDs from the welch function



# we sliced the data then fitted, rather than fitting to all data
xcutoffs = [2.5, 5, 5, 5, 5] 
ycutoffs = np.linspace(2.5, 2.8, 5)

# added reduced chi squared to the plot legend
def reduced_chisq(y_data, fit, n_params):
    n_points = len(y_data)
    residuals = fit - y_data
    return np.sum(residuals**2) / (n_points - n_params)


def make_logfun(xdata, ydata):
    """Returns a logfun for the given xdata/ydata pair."""
    def logfun1(params):
        Alpha1 = params[0]
        rolloff1 = params[1]
        FittedCurve1 = np.log(Alpha1) - np.log(rolloff1**2 + np.exp(xdata * 2))
        ErrorVector1 = FittedCurve1 - ydata
        sse1 = np.sum(ErrorVector1 ** 2)
        return sse1, FittedCurve1
    return logfun1


def fitlogcurvevarX(*args):
    """
    Input args in the form: data1, pwr1, data2, pwr2, ...
    Each `data` is a #D array with frequency in col 0, X amplitude in col 1.
    Returns (a_x, fo_x): arrays of fitted alpha and rolloff values.
    """
    pwr = [args[i] for i in range(1, len(args), 2)]
    col = 'rgbcmyk'
    a_x, fo_x = [], []

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, i in enumerate(range(0, len(args), 2)):
        data = np.array(args[i]).T   # transpose 
        data = data[1:]              # drop zero-frequency first row

        # Uncomment to fit to sliced data
        all_xdata = np.log(data[:, 0])
        mask = all_xdata < np.log(10**(xcutoffs[idx]))
        xdata = all_xdata[mask]
        all_ydata = np.log(data[:, 1])
        ydata = all_ydata[mask]

        # Uncomment to fit to all data
        # xdata = np.log(data[:, 0])
        # ydata = np.log(data[:, 1])

        # Plot all data
        xdatadis = np.log(data[:,0])
        ydatadis = np.log(data[:,1])

        logfun1 = make_logfun(xdata, ydata)

        def logfun1_sse(params):
            return logfun1(params)[0]

        start_point = np.random.rand(2)
        estimates1 = fmin(logfun1_sse, start_point, maxfun=int(1e5), maxiter=int(1e5), disp=False)

        Alpha1, rolloff1 = estimates1[0], estimates1[1]
        _, FittedCurve1 = logfun1(estimates1)

        # Reduced chi-squared (2 free parameters)
        chi2_reduced = reduced_chisq(ydata, fit=FittedCurve1, n_params=2)

        color = col[idx % 7]
        offset = idx

        ax.plot(
            np.log10(np.exp(xdatadis)),
            offset + np.log10(np.exp(ydatadis)),
            '.' + color,
            markersize=4,
            label=f'Power {pwr[idx]:.2g}'
        )
        ax.plot(
            np.log10(np.exp(xdata)),
            offset + np.log10(np.exp(FittedCurve1)),
            color,
            label=f'Fit: α={Alpha1:.4g}, f₀={rolloff1:.4g}, χ²ᵣ={chi2_reduced:.3g}'
        )

        a_x.append(Alpha1)
        fo_x.append(rolloff1)

    ax.legend()
    ax.set_xlabel('Log Frequency (Hz)')
    ax.set_ylabel('PSD (V²/Hz)')
    ax.set_title('Log Power Spectral Density (X)')
    plt.tight_layout()
    plt.show()

    return np.array(a_x), np.array(fo_x)

def fitlogcurvevarY(*args):
    """
    Input args in the form: data1, pwr1, data2, pwr2, ...
    Each `data` is a 3D array with frequency in col 0, Y amplitude in col 2.
    Returns (a_x, fo_x): arrays of fitted alpha and rolloff values.
    """
    pwr = [args[i] for i in range(1, len(args), 2)]
    col = 'rgbcmyk'
    a_x, fo_x = [], []

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, i in enumerate(range(0, len(args), 2)):
        data = np.array(args[i]).T   # transpose
        data = data[1:]              # drop zero-frequency first row

        # Uncomment to fit to sliced data
        all_xdata = np.log(data[:, 0])
        mask = all_xdata < np.log(10**(ycutoffs[idx]))
        xdata = all_xdata[mask]
        all_ydata = np.log(data[:, 2])
        ydata = all_ydata[mask]
        
        # Uncomment to fit to all data
        # xdata = np.log(data[:, 0])
        # ydata = np.log(data[:, 2])
        
        # Plot all data
        xdatadis = np.log(data[:, 0])
        ydatadis = np.log(data[:, 2])
        

        logfun1 = make_logfun(xdata, ydata)

        def logfun1_sse(params):
            return logfun1(params)[0]

        start_point = np.random.rand(2)
        estimates1 = fmin(logfun1_sse, start_point, maxfun=int(1e5), maxiter=int(1e5), disp=False)

        Alpha1, rolloff1 = estimates1[0], estimates1[1]
        _, FittedCurve1 = logfun1(estimates1)
        
        # Reduced chi-squared (2 free parameters)
        chi2_reduced = reduced_chisq(ydata, fit=FittedCurve1, n_params=2)

        color = col[idx % 7]
        offset = idx

        ax.plot(
            np.log10(np.exp(xdatadis)),
            offset + np.log10(np.exp(ydatadis)),
            '.' + color,
            markersize=4,
            label=f'Power {pwr[idx]:.2g}'
        )
        ax.plot(
            np.log10(np.exp(xdata)),
            offset + np.log10(np.exp(FittedCurve1)),
            color,
            label=f'Fit: α={Alpha1:.4g}, f₀={rolloff1:.4g}, χ²ᵣ={chi2_reduced:.3g}'
        )

        a_x.append(Alpha1)
        fo_x.append(rolloff1)

    ax.legend()
    ax.set_xlabel('Log Frequency (Hz)')
    ax.set_ylabel('PSD (V²/Hz)')
    ax.set_title('Log Power Spectral Density (Y)')
    plt.tight_layout()
    plt.show()

    return np.array(a_x), np.array(fo_x)


# Equipartition
# Make sure to divde Vx and Vy data by sensitivity obtained from the previous experiments, 
# and lower the sampling rate so that each point is independent from the last i.e. skip some points

def equipartition(xsens, ysens, *args):
    """
    Outputs the stiffness of the trap in the x and y directions.
    
    Parameters
    ----------
    xsens : float
        X sensitivity (V/um)
    ysens : float
        Y sensitivity (V/um)
    args : data1, pwr1, data2, pwr2, ...
        Each data is a 2D array, each pwr is a scalar power level (mW)
    
    Returns
    -------
    Kx, Ky : np.ndarray
        Stiffness vectors in pN/um
    """
    boltz = 1.3806503e-23
    temp = 293

    # Extract power levels from every other arg (indices 1, 3, 5, ...)
    pwr = np.array([args[i] for i in range(1, len(args), 2)], dtype=float)
    print(pwr)
    # Convert sensitivities from V/um to V/m, scaled by power
    xsens_vec = 1e6 * pwr * xsens
    ysens_vec = 1e6 * pwr * ysens

    xvariance = np.zeros(len(pwr))
    yvariance = np.zeros(len(pwr))

    for idx, i in enumerate(range(0, len(args), 2)):
        A = np.array(args[i]).T   # transpose to (rows, cols)
        xvariance[idx] = np.var(A[:235, 3] / xsens_vec[idx])  # col 4 in MATLAB = index 3
        yvariance[idx] = np.var(A[:235, 4] / ysens_vec[idx])  # col 5 in MATLAB = index 4

    # Calculate stiffness and convert to pN/um
    Kx = boltz * temp / xvariance * 1e6
    Ky = boltz * temp / yvariance * 1e6

    # Linear fit
    px = np.polyfit(pwr, Kx, 1)
    fx = np.polyval(px, pwr)
    py = np.polyfit(pwr, Ky, 1)
    fy = np.polyval(py, pwr)

    print(f"Kx fit: {px[0]:.4g}x + {px[1]:.4g}")
    print(f"Ky fit: {py[0]:.4g}x + {py[1]:.4g}")
    print(f"Kx: {Kx}")
    print(f"Ky: {Ky}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pwr, Kx, 'bo', label='X Data')
    ax.plot(pwr, fx, 'b-', label=f'X fit = {px[0]:.4g}x + {px[1]:.4g}')
    ax.plot(pwr, Ky, 'ro', label='Y Data')
    ax.plot(pwr, fy, 'r-', label=f'Y fit = {py[0]:.4g}x + {py[1]:.4g}')
    ax.set_xlabel('Power (mW)')
    ax.set_ylabel('Stiffness (pN/um)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return Kx, Ky