"""Windsor commute data — fetches departures for daily commute routes.

Two routes:
  central:   Windsor Central (WNC) → Slough (SLO) → Paddington (PAD)
             Shows next WNC→SLO, then next Elizabeth line (XR) and GWR (GW) from SLO→PAD
  riverside: Windsor Riverside (WNR) → Waterloo (WAT)
             Shows next WNR→WAT departure

Windsor branch lines (WNC→SLO, WNR→WAT) are NOT in the Darwin Kafka Push Port
feed, so those legs use the Huxley2 REST API (free, no token required).
The SLO→PAD leg uses Darwin state (Kafka feed) which carries that data.
"""

import logging

from dispatcher.apis import DispatchError, get_session

log = logging.getLogger("dispatcher")

HUXLEY_BASE = "https://huxley2.azurewebsites.net"


async def _fetch_huxley(from_crs: str, to_crs: str, count: int = 4) -> list[dict]:
    """Fetch departures from Huxley2 REST API.

    Returns list of dicts matching the commute card format:
    [{scheduled, expected, platform, status, destination, toc, cancelled}]
    """
    url = f"{HUXLEY_BASE}/departures/{from_crs}/to/{to_crs}/{count}"
    session = await get_session()
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                log.warning("Huxley2 %s→%s returned %d", from_crs, to_crs, resp.status)
                return []
            data = await resp.json()
    except Exception as e:
        log.warning("Huxley2 fetch failed for %s→%s: %s", from_crs, to_crs, e)
        return []

    services = data.get("trainServices") or []
    departures = []
    for svc in services[:count]:
        cancelled = svc.get("isCancelled", False)
        etd = svc.get("etd", "On time")
        if cancelled:
            status = "Cancelled"
        elif etd.lower() == "on time":
            status = "On Time"
        elif etd.lower() == "delayed":
            status = "Delayed"
        else:
            status = etd

        # Destination name from first destination entry
        dest_list = svc.get("destination") or []
        dest_name = dest_list[0].get("locationName", "") if dest_list else ""

        departures.append({
            "scheduled": svc.get("std", "?"),
            "expected": etd if not cancelled else "-",
            "platform": svc.get("platform") or "-",
            "status": status,
            "destination": dest_name,
            "toc": svc.get("operatorCode", ""),
            "cancelled": cancelled,
        })
    return departures


async def fetch_commute(route: str) -> dict:
    """Fetch commute data for a route ('central' or 'riverside').

    Returns dict for render_commute():
    {
        "route": "central" | "riverside",
        "sections": [
            {"label": "...", "departures": [{"scheduled", "expected", "platform", "status", "destination", "toc"}]}
        ]
    }
    """
    if route == "central":
        return await _fetch_central()
    elif route == "riverside":
        return await _fetch_riverside()
    else:
        raise DispatchError(f"Unknown commute route: {route}")


async def _fetch_central() -> dict:
    """Windsor Central → Slough → Paddington."""
    sections = []

    # Section 1: Windsor Central → Slough (Huxley2 — not in Darwin Kafka)
    wnc_deps = await _fetch_huxley("WNC", "SLO", count=3)
    sections.append({
        "label": "Windsor Central \u2192 Slough",
        "departures": wnc_deps,
    })

    # Sections 2+3: Slough → Paddington (Darwin state if available, Huxley2 fallback)
    slo_all = await _get_slo_pad_departures()
    xr_deps = [d for d in slo_all if d.get("toc") == "XR"][:3]
    sections.append({
        "label": "Slough \u2192 Paddington (Elizabeth line)",
        "toc_filter": "XR",
        "departures": xr_deps,
    })

    gw_deps = [d for d in slo_all if d.get("toc") == "GW"][:3]
    sections.append({
        "label": "Slough \u2192 Paddington (GWR)",
        "toc_filter": "GW",
        "departures": gw_deps,
    })

    return {"route": "central", "sections": sections}


async def _fetch_riverside() -> dict:
    """Windsor Riverside → Waterloo."""
    # Huxley2 — not in Darwin Kafka
    wnr_deps = await _fetch_huxley("WNR", "WAT", count=4)

    return {
        "route": "riverside",
        "sections": [
            {
                "label": "Windsor Riverside \u2192 Waterloo",
                "departures": wnr_deps,
            }
        ],
    }


async def _get_slo_pad_departures() -> list[dict]:
    """Get Slough→Paddington departures. Try Darwin state first, fall back to Huxley2."""
    # Try Darwin state (Kafka feed carries SLO→PAD data)
    try:
        from observers.darwin_consumer import get_darwin_state
        state = get_darwin_state()
        if state is not None:
            deps = state.get_departures("SLO", "PAD", count=20)
            if deps:
                return deps
    except Exception:
        pass

    # Fallback: Huxley2
    log.info("SLO→PAD: Darwin state empty, falling back to Huxley2")
    return await _fetch_huxley("SLO", "PAD", count=10)
