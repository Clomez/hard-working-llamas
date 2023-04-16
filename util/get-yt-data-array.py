import requests
import sys
import feedparser

# 

# Channel
#URL = "https://www.youtube.com/feeds/videos.xml?channel_id=UChF5O40UBqAc82I7-i5ig6A"

# Playlist
URL = "https://www.youtube.com/feeds/videos.xml?playlist_id=PLH_qQFmOkmhezyE0eBVPsajC0bJ__d7t-"


filename = "rss_feed_data_2.xml"
data_out = "data_array_politics"

data = []

def main():
    r = requests.get(URL)
    with open(filename, "wb") as f:
        f.write(r.content)
        f.close()

    o = open(filename, "r")
    a = o.read()
    o.close()

    feed = feedparser.parse(a)

    for entry in feed.entries:
        data.append("'" + str(entry.link) + "', \n" )
        # print(f"{entry.title}: {entry.link}")

    n = open(data_out, "a")
    for d in data:
        n.write(d)

    n.close()

main()