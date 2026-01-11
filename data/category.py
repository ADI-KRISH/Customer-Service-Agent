import pandas as pd
import re

filepath = r"C:/Users/GS Adithya Krishna\Desktop/internship\data/myntra_products_catalog.csv"
df = pd.read_csv(filepath)

CATEGORY_KEYWORDS = {
    "Shoes": [
        "shoe", "sneaker", "running", "trainer", "boots", "loafers", "heels",
        "flats", "sandals", "slippers", "floaters", "slides"
    ],
    "Topwear": [
        "t-shirt", "tshirt", "shirt", "top", "tee", "blouse", "hoodie",
        "sweatshirt", "kurta", "jacket", "sweater"
    ],
    "Bottomwear": [
        "jeans", "trousers", "pants", "shorts", "joggers", "leggings",
        "chinos", "trackpants", "pyjamas", "skirt"
    ],
    "Dress": [
        "dress", "gown", "frock", "maxi", "midi", "mini"
    ],
    "Innerwear": [
        "bra", "briefs", "panty", "underwear", "boxers", "trunks", "lingerie"
    ],
    "Accessories": [
        "watch", "belt", "cap", "wallet", "sunglasses", "bag", "backpack"
    ],
    "Ethnic Wear": [
        "saree", "lehenga", "kurti", "salwar", "dupatta"
    ],
    "Sportswear": [
        "tracksuit", "sports", "gym", "training", "activewear"
    ]
}
def detect_category(product_name):
    name = product_name.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for word in keywords:
            if re.search(rf"\b{word}\b", name):
                return category

    return "Other"
df["Category"] = df["ProductName"].apply(detect_category)

print(df[["ProductName", "Category"]].head(20))
output_path = r"C:/Users/GS Adithya Krishna\Desktop/internship\data/myntra_products_catalog_v2.csv"
df.to_csv(output_path, index=False)

print("✅ Category column added and saved!")
