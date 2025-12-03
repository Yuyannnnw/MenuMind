import random

FAKE_WORDS = [
    "roasted", "sweet", "potatoes", "greens", "chicken", "turkey",
    "beef", "spicy", "carrots", "salad", "tofu", "pasta",
    "beans", "rice", "broccoli", "steamed", "garlic", "herb",
    "pepper", "ginger", "sesame", "curry", "bbq"
]

def generate_fake_menu():
    k = random.randint(6, 12)
    return " ".join(random.sample(FAKE_WORDS, k))
