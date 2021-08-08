# COVID19-Retweet-UI

## Setup Instructions
To run locally:

*Python 3.7*

```
pip install -r requirements.txt

set FLASK_APP=main
set FLASK_ENV=development
flask run
```

**Navigate to http://localhost:5000 to view the GUI**



## Dataset

Parameters of the original dataset:

1. Tweet Id: Long.
2. Username: String. Encrypted for privacy issues.
3. Timestamp: Format ( "EEE MMM dd HH:mm:ss Z yyyy" ).
4. #Followers: Integer.
5. #Friends: Integer.
6. #Retweets: Integer.
7. #Favorites: Integer.
8. Entities: String. For each entity, we aggregated the original text, the annotated entity and the produced score from FEL library. Each entity is separated from another entity by char ";". Also, each entity is separated by char ":" in order to store "original_text:annotated_entity:score;". If FEL did not find any entities, we have stored "null;".
9. Sentiment: String. SentiStrength produces a score for positive (1 to 5) and negative (-1 to -5) sentiment. We splitted these two numbers by whitespace char " ". Positive sentiment was stored first and then negative sentiment (i.e. "2 -1").
10. Mentions: String. If the tweet contains mentions, we remove the char "@" and concatenate the mentions with whitespace char " ". If no mentions appear, we have stored "null;".
11. Hashtags: String. If the tweet contains hashtags, we remove the char "#" and concatenate the hashtags with whitespace char " ". If no hashtags appear, we have stored "null;".
12. URLs: String: If the tweet contains URLs, we concatenate the URLs using ":-: ". If no URLs appear, we have stored "null;"