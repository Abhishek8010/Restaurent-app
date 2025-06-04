import random
import pandas as pd
import os

menu_items = [
    "Margherita Pizza", "Pepperoni Pizza", "Veggie Pizza", "Paneer Tikka Pizza",
    "Garlic Bread", "Cheesy Garlic Bread", "French Fries", "Loaded Nachos",
    "Coke", "Pepsi", "Sprite", "Lemonade", "Cold Coffee", "Iced Tea",
    "Chocolate Brownie", "Gulab Jamun", "Vanilla Ice Cream", "Chocolate Ice Cream",
    "Pasta Alfredo", "Pasta Arrabiata", "Burger", "Cheeseburger", "Veggie Burger",
    "Grilled Sandwich", "Club Sandwich", "Chole Bhature", "Samosa", "Vada Pav",
    "Spring Roll", "Hakka Noodles", "Fried Rice", "Manchurian", "Tandoori Chicken",
    "Chicken Biryani", "Veg Biryani", "Butter Chicken", "Paneer Butter Masala",
    "Tandoori Roti", "Naan", "Stuffed Naan", "Dal Makhani", "Rajma Chawal",
    "Kheer", "Mango Lassi", "Masala Dosa", "Idli", "Medu Vada", "Upma", "Poha",
    "Dhokla", "Paneer Roll", "Chicken Roll"
]

orders = []
for _ in range(1000):
    num_items = random.randint(2, 6)
    order = random.sample(menu_items, num_items)
    orders.append(order)

df = pd.DataFrame({'items': orders})
os.makedirs("data", exist_ok=True)
df.to_csv("data/restaurant_orders.csv", index=False)
print("✅ Dataset generated and saved to data/restaurant_orders.csv")
