from flask import Flask, render_template, request, redirect, url_for, flash
from services.load import load_data, load_xgboost_feature_engineering_chart
from models.items import Item
from models.tables import SortableTable
#import plotly
#import plotly.express as px
import pandas as pd
import math
import json



app = Flask(__name__, static_folder="static")

@app.route('/<int:id>')
def flask_link(id):
    model = int(request.args.get('model', 0))
    element = Item.get_element_by_id(id, model)
    sort = request.args.get('sort', 'id')
    tab = request.args.get('tab', 'data')
    reverse = (request.args.get('direction', 'asc') == 'desc')
    # Call data from service.load with parameters
    table = SortableTable(Item.get_sorted_by(model, sort, reverse),
                          sort_by=sort,
                          sort_reverse=reverse, table_id=model)
    return render_template("index.html", load_data=load_data, table=table, item=element, tab=tab)

@app.route("/")
def index():
    sort = request.args.get('sort', 'id')
    reverse = (request.args.get('direction', 'asc') == 'desc')
    tab = request.args.get('tab', 'overview')
    model = int(request.args.get('model', 0))
    if (tab == 'data'):
        table = SortableTable(Item.get_sorted_by(model, sort, reverse),
                          sort_by=sort,
                          sort_reverse=reverse, table_id=model)
        return render_template("index.html", load_data=load_data, table=table, tab=tab)
    else:
        """
        df = pd.DataFrame({
            "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
            "Amount": [4, 1, 2, 2, 4, 5],
            "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
        })

        fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        """
        feature_engineering_chart = load_xgboost_feature_engineering_chart()


        return render_template("index.html", load_data=load_data, tab=tab, feature_engineering_chart = feature_engineering_chart)
        #return render_template("index.html", load_data=load_data, tab=tab, graphJSON=graphJSON, data = data)

if __name__ == "__main__":
  app.run(host='127.0.0.1', port=5000, debug = True)