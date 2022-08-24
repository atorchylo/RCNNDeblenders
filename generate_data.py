# Silence tqdm
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
import os
import btk
import numpy as np
from utils.ellipse_utils import moment


def generate_cosmos_HSC(
        file_path,
        num_images=200,
        batch_size=32,
        num_cpus=1,
        max_number=6,
        max_shift=5,
        stamp_size=21.6,
        valid_split=0.05,
        i_band_idx=2,
        save_path=None,
        verbose=True
):
    # set up btk generator
    catalog_paths = [file_path, file_path.replace('.fits', '_fits.fits')]
    catalog = btk.catalog.CosmosCatalog.from_file(catalog_paths, exclusion_level='none')

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
        gal_type='real',
        seed=1234
    )

    # iterate and save
    num_valid = int(round(num_images * valid_split))
    num_train = num_images - num_valid
    for name_split, num_split in zip(['valid', 'train'], [num_valid, num_train]):
        if verbose:
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
                    x1, x2 = x - Ixx ** (1/2), x + Ixx ** (1/2)
                    y1, y2 = y - Iyy ** (1/2), y + Iyy ** (1/2)
                    boxes.append((x1, y1, x2, y2))
                boxes = np.array(boxes)

                # save
                if save_path is not None:
                    directory = os.path.join(save_path, name_split)
                    image_path = os.path.join(directory, f'image_{i * batch_size + j}.npy')
                    target_path = os.path.join(directory, f'target_{i * batch_size + j}.npy')
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    np.save(image_path, image)
                    np.save(target_path, boxes)


def cpu_batch_optimization_search(file_path, save_path):
    """Simple search for optimal params"""
    import time
    results = []
    for b in [16, 32, 64, 128]:
        for n in [1, 2, 4, 8]:
            start = time.time()
            generate_cosmos_HSC(
                file_path,
                num_images=256,
                num_cpus=n,
                save_path=save_path,
                batch_size=b,
                valid_split=0.0,
                verbose=False
            )
            end = time.time()
            print(f"N_cpus = {n}, Batch_size = {b}, Time = {end - start:.3f} s")
            results.append((n, b, end - start))
    # go through the measurements and find the optimal one
    num_cpus_optimal, batch_size_optimal, min_time = results[0]
    for (n, b, t) in results[1:]:
        if t < min_time:
            min_time = t
            num_cpus_optimal = n
            batch_size_optimal = b
    print("=" * 30 + " Optimal " + "=" * 30)
    print(f"N_cpus = {num_cpus_optimal}, Batch_size = {batch_size_optimal}, Time = {min_time:.3f} s")
    return num_cpus_optimal, batch_size_optimal


if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        Generates cosmos HSC data for training. 
        """), formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to cosmos catalog main fits file, directory must contain all required files.")

    parser.add_argument("--save_path", type=str,
                        help="Directory where to save teh data. valid and train subdirectories will be added")

    parser.add_argument("--num_images", type=int, default=200,
                        help="Total number of images to generate")

    parser.add_argument("--num_cpus", type=int, default=1,
                        help="Number of CPUs to distribute across")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation")

    parser.add_argument("--optimize", type=bool, default=False,
                        help="Optimize the batch size and number of cpus for the experiment.")

    args = parser.parse_args()

    if args.optimize == True:
        num_cpus, batch_size = cpu_batch_optimization_search(args.file_path, args.save_path)
    else:
        num_cpus, batch_size = args.num_cpus, args.batch_size

    generate_cosmos_HSC(
        args.file_path,
        num_images=args.num_images,
        num_cpus=num_cpus,
        batch_size=batch_size,
        save_path=args.save_path,
        verbose=True
    )
