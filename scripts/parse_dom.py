from bs4 import BeautifulSoup

with open("agmarknet_rendered.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

print("--- SELECTS ---")
for sel in soup.find_all("select"):
    print(f"ID: {sel.get('id')} | Name: {sel.get('name')} | Class: {sel.get('class')}")
    
print("\n--- BUTTONS ---")
for btn in soup.find_all("button"):
    print(f"ID: {btn.get('id')} | Text: {btn.text.strip()} | Class: {btn.get('class')}")

print("\n--- INPUTS ---")
for inp in soup.find_all("input"):
    print(f"ID: {inp.get('id')} | Name: {inp.get('name')} | Type: {inp.get('type')} | Placeholder: {inp.get('placeholder')}")
