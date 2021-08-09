import pandas as pd
from models.model.XGBOOST import load_xgboost_prediction
from models.model.LSTM import load_lstm_prediction
from models.model.NN import load_nn_prediction
import matplotlib
from matplotlib.figure import Figure
from io import BytesIO
import base64


def load_model(model_number):
    if model_number == 0:
        return load_xgboost_prediction
    elif model_number == 1:
        return load_lstm_prediction
    elif model_number == 2:
        return load_nn_prediction
    else:
        return load_xgboost_prediction


def load_data(model_number):
    df = pd.read_feather("./data/split/data_188489.ftr")
    df_predict = pd.read_feather("./data/data_188489.ftr")

    data_dict = df.to_dict(orient="records")
    data_pred_dict = df_predict.to_dict(orient="records")

    model = load_model(model_number)
    prediction = model(df_predict)

    if model_number != 1:
        prediction = prediction[29:]

    data_dict_list = []

    # Might change according to data set provided
    data_dict = data_dict[29:]
    data_pred_dict = data_pred_dict[29:]

    for i in range(len(data_dict)):
        item = {
            "id": data_dict[i]["index"],
            "username": data_dict[i]["Username"],
            "followers": data_dict[i]["#Followers"],
            "favourites": data_dict[i]["#Favorites"],
            "friends": data_dict[i]["#Friends"],
            "retweets": data_dict[i]["#Retweets"],
            "timestamp": data_dict[i]["Timestamp"],
            "mentions_counts": data_pred_dict[i]["Mentions_count"],
            "hashtag_counts": data_pred_dict[i]["Hashtag_counts"],
            "url_counts": data_pred_dict[i]["URL_counts"],
            "predicted_retweets": int(prediction[i]),
            "model": model_number,
        }

        data_dict_list.append(item)

    return data_dict_list


def load_xgboost_feature_engineering_chart():
    feature_scores = [
        ("#Favourites", 41),
        ("#Followers", 12),
        ("confidence_mean", 12),
        ("usernamehash_col163", 10),
        ("usernamehash_col139", 8),
        ("sin_second", 8),
        ("#Friends", 7),
        ("usernamehash_col35", 6),
        ("usernamehash_col185", 5),
        ("usernamehash_col132", 4),
        ("confidence_max", 4),
        ("usernamehash_col182", 3),
        ("usernamehash_col54", 3),
        ("usernamehash_col174", 3),
        ("usernamehash_col97", 3),
        ("Sentiments1", 3),
        ("usernamehash_col102", 2),
        ("usernamehash_col50", 2),
        ("usernamehash_col73", 2),
        ("usernamehash_col56", 2),
        ("Sentiments0", 1),
        ("usernamehash_col105", 1),
        ("usernamehash_col196", 1),
        ("Hashtag_counts", 1),
        ("usernamehash_col201", 1),
        ("usernamehash_col215", 1),
    ]

    labels = [feat_score[0] for feat_score in feature_scores]
    values = [feat_score[1] for feat_score in feature_scores]

    font = {"size": 15}
    matplotlib.rc("font", **font)

    fig = Figure(figsize=(24, 8))
    ax = fig.subplots()
    ax.barh(labels, values)
    ax.set_title("feature scores")
    ax.set_xlabel("scores")
    ax.set_ylabel("feature names")
    ax.invert_yaxis()
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return data
