from datetime import date
from menu_scraper import menu_scraper

def main():
    print("=== Find Menu Page ===")
    while True:
        try:
            year = int(input("Enter year (e.g., 2025): "))
            month = int(input("Enter month (1-12): "))
            day = int(input("Enter day (1-31): "))
        except ValueError:
            print("Invalid input: please enter integers only.")
            return

        target = date(year, month, day)

        meal = input("Enter meal (Lunch/Dinner): ").strip()

        print(f"Starting scraper for {meal} on {target.isoformat()}...")
        menu = menu_scraper(target_date=target, meal=meal)

        if menu: # Success
            print("🍽️Menu: ")
            print(menu)
            break
    
    print("===  ===")



if __name__ == "__main__":
    main()
