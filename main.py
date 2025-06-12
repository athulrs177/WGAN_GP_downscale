import os
import xarray as xr
import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator
from training.train_loop import train_wgan_gp
from data.preprocessing import load_and_prepare_data

# Setup Strategy
def main():
    strategy = tf.distribute.MirroredStrategy()

    input_data_dir = 'path_to_input_data/'
    output_model_dir = 'path_to_model_output/'
    os.makedirs(output_model_dir, exist_ok=True)

    # Dataset paths
    coarse_raw = 'input_all.nc'
    fine_raw   = 'target_all_regridded.nc'

    lati, latf = 40.025, 60
    loni, lonf = -9.975, 22.025

    coarse_data = xr.open_dataset(os.path.join(input_data_dir, coarse_raw)).transpose('time','lat','lon').sel(lat=slice(lati, latf), lon=slice(loni, lonf))
    coarse_data = xr.concat([
        coarse_data.gust10, 
        coarse_data.pres_msl,
        coarse_data.u_10m, 
        coarse_data.v_10m,
        coarse_data.tot_prec,
        coarse_data.topography.expand_dims(dim={'time': coarse_data.time}),
    ], dim='features').transpose('time','lat','lon','features')

    coarse_data = coarse_data.assign_coords(features=['gust10', 'pres_msl', 'u_10', 'v_10', 'tot_prec', 'topography'])
    fine_data = xr.open_dataset(os.path.join(input_data_dir, fine_raw)).gust10.squeeze(dim='height_2',drop=True).transpose('time','lat','lon').sel(lat=slice(lati, latf), lon=slice(loni, lonf))

    dims_coarse = coarse_data.shape
    dims_fine = fine_data.shape

    train_strt, train_last = coarse_data.time[0], coarse_data.time[1199]
    test_strt, test_last = coarse_data.time[1199], coarse_data.time[-1]

    coarse_data_train = coarse_data.sel(time=slice(train_strt, train_last)).fillna(-1e-10)
    fine_data_train = fine_data.sel(time=slice(train_strt, train_last)).fillna(-1e-10)

    batch_size = 8
    z_dim = 32
    coarse_batches, fine_batches = load_and_prepare_data(coarse_data_train, fine_data_train, batch_size=batch_size)

    with strategy.scope():
        generator = build_generator(input_shape=(dims_coarse[1], dims_coarse[2], dims_coarse[3]), z_dim=z_dim)
        discriminator = build_discriminator(input_shape=(dims_fine[1], dims_fine[2], 1))

        gen_optimizer  = tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)

        generator.compile(optimizer=gen_optimizer)
        discriminator.compile(optimizer=disc_optimizer)

        print(generator.summary())
        print(discriminator.summary())

        train_wgan_gp(coarse_batches, fine_batches,
                      generator, discriminator,
                      gen_optimizer, disc_optimizer,
                      strategy=strategy,
                      total_iterations=5000, 
                      n_critic=1,
                      print_interval=5,
                      save_interval=10,
                      save_dir=output_model_dir,
                      lambda_gp=50,
                      lambda_ext=50,
                      z_dim=z_dim)

if __name__ == '__main__':
    main()