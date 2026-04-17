from glob import glob
from argparse import ArgumentParser
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-dir", type=str, default="data/ghana-satellite-imgs")
    parser.add_argument("--output-dir", type=str, default="data/features/ghana")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    files = glob(f"{args.input_dir}/**/*.ncf")
    

    dirnames = [os.path.dirname(file) for file in files]
    file_basenames = [os.path.basename(file) for file in files]

    train, test = train_test_split(
        file_basenames, stratify=dirnames, test_size=args.test_size, random_state=args.seed
    )

    ## train and test files names to different .txt files
    with open(os.path.join(args.output_dir, "train.txt"), "w") as f:
        f.write("\n".join(train))

    with open(os.path.join(args.output_dir, "test.txt"), "w") as f:
        f.write("\n".join(test))

    print(f"Splitting complete: Train files = {len(train)}\nTest files = {len(test)}")
