from flask import Flask, render_template, request, redirect, url_for, flash
from services.load import load_data, load_xgboost_feature_engineering_chart
from models.items import Item
from models.tables import SortableTable
import pandas as pd
import math
import json


app = Flask(__name__, static_folder="static")
app.secret_key = "super secret key"


@app.route("/<int:id>")
def flask_link(id):
    model = request.args.get("model", 0)
    try:
        model = int(model)
    except ValueError:
        model = 0
        flash("Model not an Integer, defaulting to first model", "error")
    sort = request.args.get("sort", "id")
    tab = request.args.get("tab", "data")
    reverse = request.args.get("direction", "asc") == "desc"
    element = Item.get_element_by_id(id, model)

    table = SortableTable(
        Item.get_sorted_by(model, sort, reverse),
        sort_by=sort,
        sort_reverse=reverse,
        table_id=model,
    )
    return render_template(
        "index.html", load_data=load_data, table=table, item=element, tab=tab
    )


@app.route("/")
def index():
    sort = request.args.get("sort", "id")
    reverse = request.args.get("direction", "asc") == "desc"
    tab = request.args.get("tab", "overview")
    model = request.args.get("model", 0)
    try:
        model = int(model)
    except ValueError:
        model = 0
        flash("Model not an Integer, defaulting to first model", "error")

    if tab == "data":
        table = SortableTable(
            Item.get_sorted_by(model, sort, reverse),
            sort_by=sort,
            sort_reverse=reverse,
            table_id=model,
        )
        return render_template("index.html", load_data=load_data, table=table, tab=tab)
    else:
        feature_engineering_chart = load_xgboost_feature_engineering_chart()

        return render_template(
            "index.html",
            load_data=load_data,
            tab=tab,
            feature_engineering_chart=feature_engineering_chart,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
