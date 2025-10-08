import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


def get_train_url(start, end, date):
    options = Options()

    driver = webdriver.Chrome(options=options)

    try:
        driver.get("https://www.bdz.bg/bg")

        cookie_button = WebDriverWait(driver, 10).until(
            expected_conditions.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    ".btn.btn-primary.btn-sm.mr-1.cw-close",
                )
            )
        )
        cookie_button.click()

        from_station = driver.find_element(By.NAME, "from")
        to_station = driver.find_element(By.NAME, "to")

        from_station.clear()
        from_station.send_keys(start)
        to_station.send_keys(end)
        # Sleep to see if the provided values are correct
        time.sleep(5)

        button = WebDriverWait(driver, 10).until(
            expected_conditions.element_to_be_clickable((By.CSS_SELECTOR, ".btn.btn-primary.btn-sm.px-3.search-submit"))
        )
        button.click()

        time.sleep(3)

        change_button = driver.find_elements(By.CSS_SELECTOR, ".text-danger.dropdown-toggle")
        for el in change_button:
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", el)
                driver.execute_script("arguments[0].click();", el)
            except Exception:
                print("Mission kaput")
    finally:
        curr_url = driver.current_url
        a = 5
        driver.close()
        return f"{curr_url}/{date}"
