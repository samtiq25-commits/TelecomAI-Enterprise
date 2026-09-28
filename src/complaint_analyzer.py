KEYWORDS={
"Internet Speed":["slow","speed","internet","data"],"Call Drops":["call drop","disconnect","dropped"],
"No Network":["no network","no service","signal"],"SMS Failure":["sms","text message"],
"SIM Problems":["sim","sim card"],"Billing":["bill","charged","billing"],"Package Issues":["package","bundle","benefits"],"Network Outage":["outage","area","unavailable"]}
def classify_complaint(text):
    t=text.lower(); scores={k:sum(w in t for w in v) for k,v in KEYWORDS.items()}; cat=max(scores,key=scores.get)
    if scores[cat]==0: cat="General Network Issue"
    if any(w in t for w in ["emergency","completely","critical","whole area","no service"]): sev="CRITICAL"
    elif any(w in t for w in ["very slow","cannot","can't","stopped","dropping"]): sev="HIGH"
    elif cat in ["No Network","Network Outage"]: sev="HIGH"
    else: sev="MEDIUM"
    return {"category":cat,"severity":sev,"keywords":[w for w in KEYWORDS.get(cat,[]) if w in t]}
