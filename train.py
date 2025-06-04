import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import os

df = pd.read_csv("data/restaurant_orders.csv")
transactions = df['items'].apply(eval).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

os.makedirs("model", exist_ok=True)
with open("rules.pkl", "wb") as f:
    pickle.dump(rules, f)

print("✅ Rules generated and saved to model/rules.pkl")
