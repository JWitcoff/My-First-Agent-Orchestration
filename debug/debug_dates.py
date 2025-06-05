# Test this function
from app.agents.utils.dates import parse_date_range_fuzzy

# Test cases
print(parse_date_range_fuzzy(["late June"], 5))
print(parse_date_range_fuzzy(["Jan 1 to Jan 8"], 7))

