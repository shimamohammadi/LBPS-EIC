import argparse
from src.data import DATA_DIR
from scripts import inference
import pandas as pd
from scripts import LBPS_EIC


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dis_dir', default='./src/data/dataset/DATASET_NAME/TEST_IMG_DIRECTORY',
                        help="Path to the distorted images")

    parser.add_argument('-ref_dir', default='./src/data/dataset/DATASET_NAME/REFERENCE_IMG_DIRECTORY',
                        help="Path to the reference images")

    parser.add_argument('-triplets_names', default='./src/data/dataset/DATASET_NAME/PAIRS/ALL_PAIRS_.CSV',
                        help="Path to a csv containing all the triplets")

    parser.add_argument('-defer_count', default=10,
                        help="Number of pairs to be deferred")

    parser.add_argument('-conditions', default=16,
                        help="Number of distorted image in TEST_IMG_DIRECTORY")

    opt = parser.parse_args()

    num_pairs = int((opt.conditions*(opt.conditions-1))/2)

    triplets_names = pd.read_csv(opt.triplets_names)

    # Inference
    mc_results = inference.MC_drop_out(
        triplets_names, opt.ref_dir, opt.dis_dir, num_pairs)

    # Return Pairs to compare
    ref_161_data = LBPS_EIC.LBPS_EIC(mc_results, opt.defer_count)

    print("Pairs to defer are:")
    print(ref_161_data.pairs_to_defer)


if __name__ == '__main__':
    main()
