from datetime import datetime, timedelta

def create_date_range(start_date, days):
    """
    Create a range of dates from start_date, incrementing by day for a given number of days.
    """

    date_range = [start_date + timedelta(days=i) for i in range(days)]

    return date_range