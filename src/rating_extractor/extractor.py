import re
import requests
import pandas as pd

@staticmethod
def get_ratings(product_url):
    r = re.search(r"i\.(\d+)\.(\d+)", product_url)
    shop_id, item_id = r[1], r[2]
    ratings_url = "https://shopee.vn/api/v2/item/get_ratings?filter=1&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"
    offset = 0
    d = {"username": [], "rating": [], "comment": []}
    i = 1;
    while True:
        data = requests.get(
            ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
        ).json()
        rating_comments_count = data['data']['item_rating_summary']['rcount_with_context']
        max_offset = rating_comments_count / 20;
        # uncomment this to print all data:
        # print(json.dumps(data, indent=4))
        # print(data["data"]["ratings"])

        if i >= max_offset or data["data"]["ratings"] is None:
            break

        for idx, rating in enumerate(data["data"]["ratings"]):
            d["username"].append(rating["author_username"])
            d["rating"].append(rating["rating_star"])
            d["comment"].append(rating["comment"])

        offset += 20
        i = i + 1

    return pd.DataFrame(d)
