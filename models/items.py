from services.load import load_data


class Item(object):
    """a little fake database"""

    def __init__(
        self,
        id,
        username,
        timestamp,
        followers,
        friends,
        favourites,
        mentions_counts,
        hashtag_counts,
        url_counts,
        retweets,
        predicted_retweets,
        model=0,
    ):
        self.id = id
        self.username = username
        self.timestamp = timestamp
        self.followers = followers
        self.friends = friends
        self.retweets = retweets
        self.favourites = favourites
        self.mentions_counts = mentions_counts
        self.hashtag_counts = hashtag_counts
        self.url_counts = url_counts
        self.predicted_retweets = predicted_retweets
        self.model = model

    @classmethod
    def get_elements(cls, model):
        if model == 0 or model == 1 or model == 2:
            items_list = [
                Item(
                    i["id"],
                    i["username"],
                    i["timestamp"],
                    i["followers"],
                    i["friends"],
                    i["favourites"],
                    i["mentions_counts"],
                    i["hashtag_counts"],
                    i["url_counts"],
                    i["retweets"],
                    i["predicted_retweets"],
                    i["model"],
                )
                for i in load_data(model)
            ]
            return items_list

        else:
            return []
            """
            return [
                Item(1, 'Z', 'zzzzz', 2, 4, 0, 0, 3, 2, 2, 3),
                Item(2, 'K', 'aaaaa', 3, 3, 1, 5, 4, 3, 1, 1),
                Item(3, 'B', 'bbbbb', 5, 1, 2, 2, 1, 0, 0, 2)
                ]
            """

    @classmethod
    def get_sorted_by(cls, model, sort, reverse=False):
        return sorted(
            cls.get_elements(model), key=lambda x: getattr(x, sort), reverse=reverse
        )

    @classmethod
    def get_element_by_id(cls, id, model):
        return [i for i in cls.get_elements(model) if i.id == id][0]
