import re
from datetime import datetime, timedelta
from typing import Tuple, List

def parse_date_range_fuzzy(dates: List[str], duration_days: int = 5) -> Tuple[str, str]:
    """
    Parse dates from various formats (exact, fuzzy terms, seasons) into
    (departure_date, return_date) in YYYY-MM-DD.
    """
    today = datetime.today()
    max_future = today + timedelta(days=365)

    def clamp(dep, ret):
        dep = max(dep, today.date())
        ret = max(ret, dep)
        dep = min(dep, max_future.date())
        ret = min(ret, max_future.date())
        return dep, ret

    # no dates → assume 30 days out for duration_days
    if not dates:
        dep, ret = clamp(today.date() + timedelta(days=30),
                         today.date() + timedelta(days=30 + duration_days))
        return str(dep), str(ret)

    # exact two-date parse
    if len(dates) == 2:
        for fmt in ("%Y-%m-%d","%m/%d/%Y","%d-%m-%Y","%B %d, %Y","%b %d, %Y"):
            try:
                d1 = datetime.strptime(dates[0], fmt).date()
                d2 = datetime.strptime(dates[1], fmt).date()
                dep, ret = clamp(d1, d2)
                return str(dep), str(ret)
            except ValueError:
                pass

    # fuzzy single term
    term = re.sub(r"[^a-z0-9 ]", "", dates[0].lower().strip())

    # handle keywords…
    if "today" in term:
        dep, ret = clamp(today.date(), today.date() + timedelta(days=duration_days))
        return str(dep), str(ret)
    if "tomorrow" in term:
        dep, ret = clamp(today.date() + timedelta(days=1),
                         today.date() + timedelta(days=1+duration_days))
        return str(dep), str(ret)
    if "next week" in term:
        next_monday = today + timedelta(days=(7 - today.weekday()))
        dep, ret = clamp(next_monday.date(), next_monday.date() + timedelta(days=duration_days))
        return str(dep), str(ret)
    if "next month" in term:
        year = today.year + (1 if today.month == 12 else 0)
        month = 1 if today.month == 12 else today.month+1
        dep_raw = datetime(year, month, 15).date()
        dep, ret = clamp(dep_raw, dep_raw + timedelta(days=duration_days))
        return str(dep), str(ret)

    # early/mid/late month & seasons…
    month_map = {
      **{m:i for i,m in enumerate(["january","february","march","april","may","june",
                                   "july","august","september","october","november","december"], start=1)},
      **{m[:3]:i for m,i in zip(
         ["january","february","march","april","may","june",
          "july","august","september","october","november","december"],
         range(1,13)
      )}
    }

    # early/mid/late month
    for key in ("early","mid","late"):
        if key in term:
            for name,mon in month_map.items():
                if name in term:
                    day = {"early":5,"mid":15,"late":25}[key]
                    year = today.year + (1 if today.month>mon else 0)
                    dep_raw = datetime(year,mon,day).date()
                    dep,ret = clamp(dep_raw, dep_raw+timedelta(days=duration_days))
                    return str(dep),str(ret)

    # season
    seasons = {"winter":(1,15),"spring":(4,15),"summer":(7,15),"autumn":(10,15),"fall":(10,15)}
    for season,(mon,day) in seasons.items():
        if season in term:
            year = today.year + (1 if today.month>mon else 0)
            dep_raw = datetime(year,mon,day).date()
            dep,ret = clamp(dep_raw, dep_raw+timedelta(days=duration_days))
            return str(dep),str(ret)

    # fallback default
    dep, ret = clamp(today.date()+timedelta(days=30),
                     today.date()+timedelta(days=30+duration_days))
    return str(dep), str(ret)
