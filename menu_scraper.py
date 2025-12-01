from datetime import date, datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains

import time

URL = "https://virginia.campusdish.com/en/locationsandmenus/observatoryhilldiningroom/"

def _month_key(dt: datetime) -> int:
    return dt.year * 12 + dt.month

def _get_visible_year_month(driver) -> datetime:
    month_el = driver.find_element(By.CSS_SELECTOR, ".react-datepicker__month[aria-label^='month']")
    ym = month_el.get_attribute("aria-label").split()[-1].strip()  # e.g., '2025-11'
    return datetime.strptime(ym, "%Y-%m")

def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def _select_date_forward_only(driver, target: date) -> bool:
    # Open the date picker
    date_btn = driver.find_element(By.CSS_SELECTOR, "button#aria-date-controller")
    driver.execute_script("arguments[0].click();", date_btn)
    print("Opened date picker.")
    time.sleep(1)

    current = _get_visible_year_month(driver)
    t_month = datetime(target.year, target.month, 1)

    if _month_key(t_month) < _month_key(current):
        print("Target month is in the past relative to the visible calendar. Not navigating backward.")
        return False

    # Find the target date
    while _month_key(current) < _month_key(t_month):
        next_btn = driver.find_element(
            By.CSS_SELECTOR,
            "button[aria-label='Next Month'].react-datepicker__navigation--next"
        )
        driver.execute_script("arguments[0].click();", next_btn)
        time.sleep(1)
        current = _get_visible_year_month(driver)

    aria_label_exact = f"Choose {target.strftime('%A')}, {target.strftime('%B')} {_ordinal(target.day)}, {target.year}"
    xpath_exact = ("//div[contains(@class,'react-datepicker__day') and "
                   f"@aria-label=\"{aria_label_exact}\"]")

    xpath_fallback = (
        "//div[contains(@class,'react-datepicker__day') and "
        f"contains(@aria-label, '{target.strftime('%B')} {_ordinal(target.day)}, {target.year}') and "
        "not(contains(@class,'outside-month'))]"
    )

    try:
        try:
            day_el = driver.find_element(By.XPATH, xpath_exact)
        except Exception:
            day_el = driver.find_element(By.XPATH, xpath_fallback)
    except Exception:
        print(f"🚫 Could not locate day element for {target}. Possibly outside displayed range.")
        return False

    # ✅ Check for disabled / unavailable states
    is_disabled = day_el.get_attribute("aria-disabled") == "true"
    span_text = ""
    try:
        span = day_el.find_element(By.TAG_NAME, "span")
        span_text = span.get_attribute("title") or span.get_attribute("aria-label") or ""
    except Exception:
        pass

    if is_disabled or "No Menu" in span_text or "unavailable" in span_text.lower():
        print(f"🚫 The date {target.isoformat()} is unavailable (No Menu). Please choose another date.")
        return False

    # Otherwise, click it
    driver.execute_script("arguments[0].click();", day_el)
    print(f"✅ Selected {target.isoformat()}")
    time.sleep(1)
    return True

def _select_meal(driver, meal: str) -> bool:
    selectors_to_try = [
        "input#aria-meal-input[role='combobox']",
        ".css-geczwp-indicatorContainer",   # chevron container
        ".css-18hlnx5",                     # right-side indicator wrapper
        ".css-6e0f30-control",              # whole control
    ]

    # Try to find the input (we'll type into this)
    try:
        input_el = driver.find_element(By.CSS_SELECTOR, "input#aria-meal-input[role='combobox']")
    except Exception:
        print("⚠️ Could not locate the meal combobox input.")
        return False

    # Make sure it’s on screen
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", input_el)
    except Exception:
        pass
    time.sleep(0.3)

    # OPEN the dropdown
    for sel in selectors_to_try:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            try:
                el.click()
            except Exception:
                driver.execute_script("arguments[0].click();", el)
            time.sleep(0.4)
            try:
                expanded = input_el.get_attribute("aria-expanded")
                if expanded and expanded.lower() == "true":
                    break
            except Exception:
                pass
        except Exception:
            continue

    # Type the meal and press Enter (React-Select filters options)
    try:
        input_el.send_keys(meal)
        time.sleep(0.5) 
        input_el.send_keys(Keys.ENTER)
        time.sleep(0.8)
    except Exception:
        print("🚫 Typing/selection failed in the meal dropdown.")
        return False

    print(f"🍽️ Selected meal: {meal}")
    return True

def scrape_station(driver, station_name):
        """Return dish names from one station."""
        station = driver.find_element(
            By.XPATH,
            f"//div[contains(@class,'MenuStation_no-categories') and .//h2[normalize-space()='{station_name}']]"
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", station)
        time.sleep(0.8)  # allow lazy loading

        menu = []
        cards = station.find_elements(By.CSS_SELECTOR, "li[data-testid='product-card']")
        for card in cards:
            name_els = card.find_elements(By.CSS_SELECTOR, "[data-testid='product-card-header-link']")
            if not name_els:
                name_els = card.find_elements(By.CSS_SELECTOR, "[data-testid='product-card-header-title']")
            if name_els:
                menu.append(name_els[0].text.strip())

        return menu


def menu_scraper(target_date: date, meal: str):
    """
    Launches Selenium, opens the CampusDish O-Hill page,
    clicks 'Change', then selects the given target_date (forward-only).
    Extend this function to select meal & scrape items afterward.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(45)

    try:
        # Open page
        driver.get(URL)
        time.sleep(1.5)

        # Click "Change" to open dialog
        change_btn = driver.find_element(By.CSS_SELECTOR, "button.DateMealFilterButton")
        driver.execute_script("arguments[0].click();", change_btn)
        print("Opened Change dialog.")
        time.sleep(1)

        # Enforce forward-only policy
        if target_date < date.today():
            print(f"Target date {target_date} is before today. Aborting per forward-only policy.")
            return

        if not _select_date_forward_only(driver, target_date):
            return
        
        if not _select_meal(driver, meal):
            return
        
        # Click "Done" button to update the menu
        try:
            btn = driver.find_element(By.XPATH, "//button[.//span[normalize-space()='Done']]")
            driver.execute_script("arguments[0].click();", btn)
            print("✅ Clicked Done.")
            time.sleep(4.0)
        except Exception:
            print("⚠️ Done button not found.")
            return
        
        # Scrape the menu from Hearth, True Balance, and Meze
        menu = {}
        for station in ["Hearth", "True Balance"]:
            try:
                menu[station] = scrape_station(driver, station)
            except Exception as e:
                print(f"Could not scrape {station}: {e}")
                menu[station] = []
        return menu
    
    finally:
        driver.quit()

