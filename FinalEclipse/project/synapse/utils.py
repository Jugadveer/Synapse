def get_risk_level(result):
    r = result.lower()

    if "no impairment" in r:
        return "LOW"
    elif "very mild" in r:
        return "MEDIUM"
    else:
        return "HIGH"