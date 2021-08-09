from flask_table import Table, Col, LinkCol
from flask import url_for, request


class SortableTable(Table):
    css_properties = {
        "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
    }

    id = Col(
        "ID",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    username = Col(
        "Username",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    timestamp = Col(
        "Timestamp",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    followers = Col(
        "#Followers",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    friends = Col(
        "#Friends",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    retweets = Col(
        "Retweets",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    favourites = Col(
        "#Favourites",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    mentions_counts = Col(
        "Mentions_counts",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    hashtag_counts = Col(
        "Hashtag_counts",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    url_counts = Col(
        "URL_counts",
        column_html_attrs={
            "class": "text-center border text-xs md:text-sm lg:text-base xl:text-sm"
        },
    )
    link = LinkCol(
        "View",
        "flask_link",
        url_kwargs=dict(id="id", model="model"),
        url_kwargs_extra=dict(tab="data"),
        allow_sort=False,
        column_html_attrs={
            "class": "text-center text-xs md:text-sm lg:text-base xl:text-sm text-blue-700"
        },
    )
    allow_sort = True
    classes = ["table-auto", "border-2", "border-black", "border-collapse"]

    def sort_url(self, col_key, reverse=False):
        if reverse:
            direction = "desc"
        else:
            direction = "asc"
        return url_for(
            "index", sort=col_key, direction=direction, tab="data", model=self.table_id
        )
