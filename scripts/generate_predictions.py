import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from glob import glob
import os
from tqdm import tqdm


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
    pred_probs = pipeline.predict_proba(X)
    return pd.DataFrame({"preds": preds})


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


if __name__ == "__main__":
    args = parse_args()
    model = select_model(args.model_id)
    pipeline = get_pipeline(model)

    train_files, test_files = get_train_test_files(args.split_dir)


    files = glob(f"{args.input_dir}/*.csv")
    for file in tqdm(files):
        df = pd.read_csv(file)

        df_train, df_test = get_train_test_split(df, train_files, test_files)

        ## getting train and test splits
        X_train, y_train = df_train.filter(regex="^feature"), df_train["label"]
        X_test, y_test = df_test.filter(regex="^feature"), df_test["label"]

        ## fitting the pipeline
        pipeline.fit(X_train, y_train)

        ## predicting
        preds_df = predict(pipeline, X_test)
        preds_df["image_name"] = df_test["image_name"]
        preds_df["target"] = y_test

        output_dir = os.path.join(args.output_dir, args.model_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        output_file = os.path.join(output_dir, os.path.basename(file))  
        preds_df.to_csv(output_file, index=False)
