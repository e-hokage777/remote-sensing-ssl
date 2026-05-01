import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from glob import glob
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_train_test_split(df, train_files, test_files):
    base_names = df["image_name"].str.split("/").str[-1]

    df_train = df[base_names.isin(train_files)]
    df_test = df[base_names.isin(test_files)]

    # X_train = df_train.filter(regex="^feature")
    # y_train = df_train["label"]

    # X_test = df_test.filter(regex="^feature")
    # y_test = df_test["label"]

    # return X_train, X_test, y_train, y_test
    return df_train, df_test


def get_train_test_files(path):
    train_files = []
    test_files = []
    with open(os.path.join(path, "train.txt"), "r") as f:
        train_files = f.read().splitlines()
    with open(os.path.join(path, "test.txt"), "r") as f:
        test_files = f.read().splitlines()
    return train_files, test_files


def get_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def predict(pipeline, X):
    preds = pipeline.predict(X)
    # pred_probs = pipeline.predict_proba(X)
    return pd.DataFrame({"preds": preds})


def create_confusion_matrix(y_true, y_pred, label_map, save_path):
    label_names = [label_map[k] for k in sorted(label_map)]
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    y_true_names = [label_names[k] for k in y_true]
    y_pred_names = [label_names[k] for k in y_pred]

    disp = ConfusionMatrixDisplay.from_predictions(y_true_names, y_pred_names, colorbar=False, cmap="Blues")
    # disp.plot()
    disp.ax_.tick_params(axis="x", rotation=90, labelsize=8)
    disp.ax_.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(save_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-id", type=str, default="random_forset")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def select_model(model_id):
    if model_id == "random_forset":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier()
    
    if model_id == "svm":
        from sklearn.svm import SVC

        return SVC()
    
    if model_id == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier()
    
    if model_id == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression()


if __name__ == "__main__":
    args = parse_args()
    model = select_model(args.model_id)
    pipeline = get_pipeline(model)

    train_files, test_files = get_train_test_files(args.split_dir)

    files = glob(f"{args.input_dir}/*.csv")
    for file in tqdm(files, unit="file", desc="Generating predictions and plots"):
        df = pd.read_csv(file)

        df_train, df_test = get_train_test_split(df, train_files, test_files)
        df_train.reset_index(inplace=True)
        df_test.reset_index(inplace=True)

        ## getting train and test splits
        X_train, y_train = df_train.filter(regex="^feature"), df_train["label"]
        X_test, y_test = df_test.filter(regex="^feature"), df_test["label"]

        ## fitting the pipeline
        pipeline.fit(X_train, y_train)

        ## creating label map
        label_names = df_test["image_name"].str.split("/").str[-2]
        label_map = (
            df_test[["label", "label_names"]]
            .drop_duplicates()
            .set_index("label")
            .to_dict()
        )["label_names"]
        label_map = {int(k): v.replace("_ghana", "") for k, v in label_map.items()}

        ## predicting
        preds_df = predict(pipeline, X_test)
        preds_df["image_name"] = df_test["image_name"]
        preds_df["label_names"] = df_test["image_name"].str.split("/").str[-2]
        preds_df["target"] = y_test

        output_dir = os.path.join(args.output_dir, args.model_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ## creating confusion matrix
        fig_output_path = os.path.join(
            "plots/evaluations", args.model_id, os.path.basename(file) + f"-{args.model_id}.png"
        )
        create_confusion_matrix(y_test, preds_df["preds"], label_map, fig_output_path)

        output_file = os.path.join(output_dir, os.path.basename(file))
        preds_df.to_csv(output_file, index=False)
