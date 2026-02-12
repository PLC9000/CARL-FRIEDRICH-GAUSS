import datetime


def parse_period(
    last_n_days: int | None,
    start: str | None,
    end: str | None,
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return (start_dt, end_dt) from either last_n_days or explicit start/end strings."""
    now = datetime.datetime.utcnow()
    if last_n_days is not None:
        end_dt = now
        start_dt = now - datetime.timedelta(days=last_n_days)
        return start_dt, end_dt

    if start is None or end is None:
        raise ValueError("Provide either last_n_days or both start and end dates")

    try:
        start_dt = datetime.datetime.fromisoformat(start)
    except ValueError:
        raise ValueError(f"Invalid start date format: {start}")
    try:
        end_dt = datetime.datetime.fromisoformat(end)
    except ValueError:
        raise ValueError(f"Invalid end date format: {end}")

    if start_dt >= end_dt:
        raise ValueError("start must be before end")
    if (end_dt - start_dt).days > 365:
        raise ValueError("Time range must not exceed 365 days")

    return start_dt, end_dt
