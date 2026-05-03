import pandas as pd
from argparse import ArgumentParser
import os
from glob import glob
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="predictions")
    parser.add_argument("--output-dir", type=str, default="metrics")

    return parser.parse_args()


def compute_kappas(df):
    return cohen_kappa_score(df["target"], df["preds"])


def compute_class_accuracies(df):

    preds = df["preds"]
    targets = df["target"]

    cm = confusion_matrix(targets, preds)
    accuracy = cm.diagonal() / cm.sum(axis=1)
    return accuracy


if __name__ == "__main__":
    args = parse_args()

    ## creating the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    all_accuracies = dict()
    kappa_coefs = dict()

    model_names = [os.path.basename(path) for path in glob(f"{args.input_dir}/*")]

    label_map = None

    for model_name in model_names:
        files = glob(f"{args.input_dir}/{model_name}/*.csv")

        for file in files:
            df = pd.read_csv(file)

            ## creating the label map if it doesn't already exist
            if label_map is None:
                label_map = df.drop_duplicates(subset=["target"])[
                    ["target", "label_names"]
                ]
                label_map = label_map.set_index("target")["label_names"]
                label_map.sort_index(inplace=True)

            filename = os.path.basename(file).split(".")[0]
            _, ssl_model, dataset = filename.split("-")

            ## computing accuracies
            accuracies = compute_class_accuracies(df)
            all_accuracies[(dataset, ssl_model, model_name)] = accuracies

            ## computing kappas
            kappas = compute_kappas(df)
            kappa_coefs[(ssl_model, dataset)] = {model_name: kappas}

    accuracies_df = pd.DataFrame(all_accuracies, index=label_map)
    accuracies_df.to_csv(f"{args.output_dir}/class_accuracies.csv")

    kappa_df = pd.DataFrame(kappa_coefs, index=model_names)
    kappa_df.to_csv(f"{args.output_dir}/kappas.csv")

    # all_accuracies = pd.DataFrame.from_dict(all_accuracies, orient="index").reset_index()
    # all_accuracies.columns = ["model_name", "ssl_model", "dataset", "accuracy"]

    # kappa_coefs = pd.DataFrame.from_dict(kappa_coefs, orient="index").reset_index()
    # kappa_coefs.columns = ["model_name", "ssl_model", "dataset", "kappa"]

    # all_accuracies.to_csv(f"{args.output_dir}/class_accuracies.csv", index=False)
    # kappa_coefs.to_csv(f"{args.output_dir}/kappas.csv", index=False)
