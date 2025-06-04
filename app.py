import streamlit as st
st.set_page_config(page_title="🍽️ Restaurant Menu Combo Builder", layout="wide")

import pandas as pd
import pickle
import subprocess
import os
from mlxtend.frequent_patterns import association_rules

# Load rules
@st.cache_data
def load_rules():
    with open("rules.pkl", "rb") as f:
        return pickle.load(f)

rules = load_rules()

# Extract all menu items
all_items = set()
if not rules.empty:
    for itemset in rules['antecedents']:
        all_items.update(list(itemset))
    for itemset in rules['consequents']:
        all_items.update(list(itemset))

all_items = sorted(all_items)

# Sidebar Inputs
st.sidebar.header("📋 Choose Menu Items")
required_item = st.sidebar.selectbox("🍔 Required Item", [""] + all_items)
optional_item_1 = st.sidebar.selectbox("🍷 Optional Item 1", [""] + all_items)
optional_item_2 = st.sidebar.selectbox("🍰 Optional Item 2", [""] + all_items)

# Collect selected items
selected_items = []
if required_item: selected_items.append(required_item)
if optional_item_1: selected_items.append(optional_item_1)
if optional_item_2: selected_items.append(optional_item_2)

# Filter function - updated to handle partial matches
def find_combos(rules, items):
    if not items:
        return pd.DataFrame()  # No input, return empty DataFrame
    
    # Create a mask for rules where at least one selected item is in antecedents
    mask = rules['antecedents'].apply(lambda x: any(item in x for item in items))
    
    # Filter rules and sort by confidence and lift
    filtered = rules[mask].copy()
    if not filtered.empty:
        # Calculate match score (number of selected items in antecedents)
        filtered['match_score'] = filtered['antecedents'].apply(
            lambda x: len(set(items) & set(x)))
        
        # Sort by match score (higher first), then confidence (higher first), then lift (higher first)
        filtered = filtered.sort_values(
            by=['match_score', 'confidence', 'lift'], 
            ascending=[False, False, False]
        ).head(5)
    
    return filtered

# Main Page
st.title("🍽️ Restaurant Menu Combo Builder")
st.markdown("Use association rules to get food combo recommendations!")

# Recommended Combos
st.subheader("🎯 Recommended Combos")
if selected_items:
    result = find_combos(rules, selected_items)
    if not result.empty:
        for _, row in result.iterrows():
            antecedents = ", ".join(list(row['antecedents']))
            consequent = ", ".join(list(row['consequents']))
            confidence = row['confidence']
            lift = row['lift']
            
            # Highlight the items that match user's selection
            matched_items = set(selected_items) & set(row['antecedents'])
            highlighted_antecedents = antecedents
            for item in matched_items:
                highlighted_antecedents = highlighted_antecedents.replace(
                    item, f"**{item}**")
            
            st.markdown(
                f"**If you order:** `{highlighted_antecedents}`  \n"
                f"**Recommended:** `{consequent}`  \n"
                f"🎯 Confidence: {confidence:.2f}, 🔗 Lift: {lift:.2f}"
            )
    else:
        st.warning("No strong combos found for selected items. Try different items.")
else:
    st.info("👉 Please select at least one menu item.")

# Rules Table
st.subheader("📊 Association Rules")
if not rules.empty:
    display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ", ".join(list(x)))
    display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ", ".join(list(x)))
    st.dataframe(display_rules)
else:
    st.warning("⚠️ No rules found. Please retrain the model.")

# Download Rules
csv = rules.to_csv(index=False)
st.download_button("⬇️ Download Rules as CSV", csv, "rules.csv", "text/csv")

# Retrain Button
if st.button("🔁 Retrain Model"):
    with st.spinner("Retraining..."):
        subprocess.run(["python", "train.py"])
        st.success("✅ Model retrained successfully! Please refresh the page.")
