import os
import btk
import numpy as np
from src.utils.ellipse_utils import moment

COSMOS_CATALOG_PATHS = [
    'raw_data/catalogs/cosmos/real_galaxy_catalog_26_extension_example.fits',
    'raw_data/catalogs/cosmos/real_galaxy_catalog_26_extension_example_fits.fits',
]


def generate_cosmos_HSC(
        num_galaxies=200,
        batch_size=32,
        num_cpus=1,
        max_number=3,
        max_shift=7,
        stamp_size=21.6,
        valid_split=0.05,
        save_path='raw_data',
        i_band_idx=2,
):
    # set up btk generator
    catalog = btk.catalog.CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, exclusion_level='none')
    survey = btk.survey.get_surveys('HSC')
    sampling_function = btk.sampling_functions.DefaultSampling(
        stamp_size=stamp_size, max_number=max_number, maxshift=max_shift)
    draw_generator = btk.draw_blends.CosmosGenerator(
        catalog,
        sampling_function,
        survey,
        batch_size=batch_size,
        stamp_size=stamp_size,
        cpus=num_cpus,
        add_noise='all',
        verbose=False,
        gal_type='real',
    )

    # iterate and save
    num_valid = int(round(num_galaxies * valid_split))
    num_train = num_galaxies - num_valid
    for name_split, num_split in zip(['valid', 'train'], [num_valid, num_train]):
        print(f"Generating {num_split} images for {name_split}")
        for i in range(0, num_split, batch_size):
            data = next(draw_generator)
            for j in range(batch_size):
                image = data['blend_images'][j]
                isolated = data['isolated_images'][j]
                blend_list = data['blend_list'][j]

                # generate boxes
                boxes = []
                for idx in range(len(blend_list)):
                    # compute moments
                    galaxy = isolated[idx, i_band_idx]
                    Ixx = moment(galaxy, 2, 0)
                    Iyy = moment(galaxy, 0, 2)
                    # get box coordinates
                    x, y = blend_list[idx]['x_peak', 'y_peak']
                    x1, x2 = x - Ixx, x + Ixx
                    y1, y2 = y - Iyy, y + Iyy
                    boxes.append((x1, y1, x2, y2))
                boxes = np.array(boxes)

                # save
                directory = os.path.join(save_path, name_split)
                image_path = os.path.join(directory, f'image_{i * batch_size + j}.npy')
                target_path = os.path.join(directory, f'target_{i * batch_size + j}.npy')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                np.save(image_path, image)
                np.save(target_path, boxes)


if __name__ == '__main__':
    generate_cosmos_HSC(
        num_galaxies=200,
        batch_size=32,
        num_cpus=1,
        max_number=6,
        max_shift=5,
        stamp_size=21.6,
        valid_split=0.05,
        save_path='raw_data/cosmos_HSC',
        i_band_idx=2,
    )