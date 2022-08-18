"""Quick and dirty"""
import btk
stamp_size = 21.6
COSMOS_CATALOG_PATHS = [
    'raw_data/catalogs/cosmos/real_galaxy_catalog_26_extension_example.fits',
    'raw_data/catalogs/cosmos/real_galaxy_catalog_26_extension_example_fits.fits',
]
catalog = btk.catalog.CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, exclusion_level='none')
survey = btk.survey.get_surveys('HSC')

sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size, max_number=6, maxshift=7)

draw_generator = btk.draw_blends.CosmosGenerator(
        catalog,
        sampling_function,
        survey,
        batch_size=50,
        stamp_size=stamp_size,
        cpus=1,
        add_noise='all',
        verbose=False,
        gal_type='real',
        save_path='raw_data'
    )

if __name__ == '__main__':
    batch = next(draw_generator)