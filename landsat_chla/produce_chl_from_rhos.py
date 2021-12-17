import joblib
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from ncwrite import nc_write
from netCDF4 import Dataset
from sklearn.ensemble import RandomForestRegressor


matplotlib.use('agg')

dict_waves = {'L5': [486, 571, 660, 839, 1678, 2217],
                'L7': [479, 561, 661, 835, 1650, 2208],
                'L8': [483, 561, 655, 865, 1609, 2201]}


def get_retrieval(l2file, dt):
    nc = Dataset(l2file, 'r')
    data = nc[dt][:]
    nc = None
    return data


def plot_chl(nc_file):
    png_file = os.path.splitext(nc_file)[0] + '_chl.png'
    if not os.path.exists(png_file):
        chl = get_retrieval(nc_file, 'chla')   
        plt.figure()
        plt.imshow(chl, vmin=0, vmax=60, cmap='jet')
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar(ticks=range(0, 61, 20), shrink=.95)
        cbar.minorticks_off()
        cbar.set_label('Chla ($\mu$g/L)', fontsize=12, fontname='Arial')
        plt.tight_layout()
        plt.savefig(png_file, dpi=300)
    return


def imread_navigation(l2path):
    nc = Dataset(l2path, 'r')
    lat = nc['lat'][:]
    lon = nc['lon'][:]
    nc = None
    return lat, lon


def imread_l2(l2path):
    mission = os.path.basename(l2path).split('_')[0]
    assert mission in dict_waves.keys(), 'Only support L5TM, L7ETM, L8OLI'
    waves = dict_waves[mission]
    nc = Dataset(l2path, 'r')
    nwaves = len(waves)
    for i, wave in enumerate(waves):
        band = nc['rhorc_' + str(int(wave))][:]
        if i == 0:
            data = np.zeros((band.shape[0], band.shape[1], len(waves)))
        data[:, :, i] = band
    nc = None
    # make a quick AC by substracting rhorc_SWIR2
    data_ac = np.zeros((data.shape[0], data.shape[1], nwaves-1))
    for i in range(nwaves-1):
        data_ac[:, :, i] = data[:, :, i] - data[:, :, -1]
    # fill neg value to nan
    data_ac[data_ac <= 0] = np.nan
    # remove clouds and lands using rhos_1600 > 0.0215
    return data_ac


def add_ratios(input_x):
    # add extenal spectral index to rrc data
    bgi = input_x[:, :, 0] / input_x[:, :, 1]
    rbi = input_x[:, :, 2] / input_x[:, :, 0]
    ngi = input_x[:, :, 3] / input_x[:, :, 1]
    nri = input_x[:, :, 3] / input_x[:, :, 2]
    fai = input_x[:, :, 3]-input_x[:, :, 2]+(input_x[:, :, 4]-input_x[:, :, 2])*(865.0-655.0)/(1609.0-655.0)
    exp_index = (np.exp(input_x[:, :, 3])-np.exp(input_x[:, :, 2]))/(np.exp(input_x[:, :, 3])+np.exp(input_x[:, :, 2]))
    ratios = np.dstack((bgi, rbi, ngi, nri, fai, exp_index))
    input_var = np.append(input_x, ratios, axis=2)
    return input_var


def image_estimate(data, model, x_scaler, y_scaler):
    """Takes any number of input bands (shaped [Height, Width]) and
    returns the products for that image, in the same shape."""
    expected_features = 11
    assert (data.shape[-1] == expected_features), (
        f'Got {data.shape[-1]} features; expected {expected_features} features for Landsat sensor')
    im_shape = data.shape[:-1]
    im_data = data.reshape((-1, expected_features))
    x_test = x_scaler.transform(im_data)
    # fill nan to 0
    x_test = np.nan_to_num(x_test)
    # inversion
    y_hat = model.predict(x_test).reshape(-1, 1)
    est = np.exp(y_scaler.inverse_transform(y_hat))
    # reshape to 2D array
    chl = est.reshape(im_shape)
    # set retrievals with nan inputs to nan
    # gen a array of nan, if any pixel has nan in axis =1, the sum would be nan
    # data_2d_sum = np.sum(data, axis=2)
    # for all pixels would mask most of data, the mask would be processed in postprocessing
    chl[chl < 0] = np.nan
    return chl


def mask_algae(nc_file):
    nc = Dataset(nc_file, 'r')
    lat = nc['algae'][:]
    nc = None
    return lat, lon

###-----------------------------------------MAIN-----------------------------------------####
# load model and scalers
rf_model = joblib.load('rf_chl_rrc.json')
x_scaler = joblib.load('x_scaler.json')
y_scaler = joblib.load('y_scaler.json')
print('RF model and required scalers have been loaded.')
# load image data
l2_dir = r'K:\Landsat_L2\taihu'
out_path = r'D:\Taihu_Chl_landsat\retrievals'
files = glob(os.path.join(l2_dir, '*L2R.nc'))
print('{} files will be processed...'.format(len(files)))

for i, filename in enumerate(files):
    nc_file = os.path.join(out_path, os.path.basename(
    filename).split('.')[0] + '_chl.nc')
    if not os.path.exists(nc_file):
        rrc_data = imread_l2(filename)
        print(('{}/{} : ' + os.path.basename(filename) + ' readed...').format(i+1, len(files)))
        # add ratios
        data = add_ratios(rrc_data)
        # estimate
        chl = image_estimate(data, rf_model, x_scaler, y_scaler)
        # remove the retrivals over cloud, mask, and thick algae scums
        chl[np.isnan(rrc_data[:, :, -2])] = np.nan
        print('  >>> Chla have been estimated.')
    # write result with navigation data
        nc_write(nc_file, 'chla', chl, new=True)
        lat, lon = imread_navigation(filename)
        nc_write(nc_file, 'latitude', lat, new=False)
        nc_write(nc_file, 'longitude', lon, new=False)
        print('  >>> {} has been written.'.format(nc_file))
    # plot chl
    png_file = os.path.splitext(nc_file)[0] + '.png'
    plot_chl(nc_file)
    print('  >>> Chla plots have been estimated.')
