from runtime.tool_belt import search_web, scrape_page, python_repl
from runtime.stop_condition import evaluate_stop_condition
from spec import StopConditionSpec

def run_tests():
    print("--- 1. Testing python_repl ---")
    out = python_repl("print('Hello from sandbox!')\nfor i in range(3): print(i)")
    print("Output:\n" + out.strip())

    print("\n--- 2. Testing scrape_page ---")
    # Wikipedia is usually reliable and fast
    page_text = scrape_page("https://en.wikipedia.org/wiki/Main_Page")
    print("Scraped text preview (first 150 chars):")
    print(page_text[:150].replace('\n', ' '))

    print("\n--- 3. Testing search_web (DuckDuckGo) ---")
    search_res = search_web("JetBrains IDEs")
    print("Search results preview (first 150 chars):")
    print(search_res[:150].replace('\n', ' '))

    print("\n--- 4. Testing stop_condition evaluations ---")
    cond = StopConditionSpec(min_report_length=50, must_include_citations=True, max_steps=10)
    
    # Test short report
    is_valid, msg = evaluate_stop_condition("Too short.", cond)
    print("Short report:", is_valid, "| Msg:", msg)

    # Test missing citation
    long_text = "This is a much longer report that exceeds the fifty character minimum requirement but has no citations."
    is_valid, msg = evaluate_stop_condition(long_text, cond)
    print("Missing citation:", is_valid, "| Msg:", msg)

    # Test valid report
    valid_text = long_text + " Source: https://jetbrains.com"
    is_valid, msg = evaluate_stop_condition(valid_text, cond)
    print("Valid report:", is_valid, "| Msg:", msg)

if __name__ == "__main__":
    run_tests()
