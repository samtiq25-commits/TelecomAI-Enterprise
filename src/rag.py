from .config import KB_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
DOCS = {
    "network_congestion.txt": (
        "Network congestion causes increased latency, packet loss "
        "and degraded throughput. Check traffic trends, neighboring "
        "towers and backhaul utilization before escalation."
    ),

    "high_latency.txt": (
        "For high latency, check traffic, packet loss, backhaul links, "
        "neighboring towers, CPU and memory, plus recent maintenance "
        "or configuration events."
    ),

    "equipment_failure.txt": (
        "Equipment degradation indicators include abnormal temperature, "
        "high CPU, increasing error counts, long uptime and insufficient "
        "maintenance. Schedule preventive maintenance for high-risk equipment."
    ),

    "customer_support_sop.txt": (
        "Classify the complaint, estimate severity, identify location "
        "and service type, then correlate the complaint with network health. "
        "Do not expose internal infrastructure details without authorization."
    ),

    "account_balance.txt": (
        "To check account balance, customers can use the telecom provider's "
        "official mobile application, USSD balance inquiry code, or customer "
        "support. Prepaid customers should check their remaining account "
        "credit, data allowance, and validity. The exact USSD code depends "
        "on the telecom provider."
    ),

    "package_management.txt": (
        "To change a mobile package, customers can use the provider's "
        "official mobile application, official USSD package menu, or "
        "customer support. Before activating a package, verify its price, "
        "data allowance, call minutes, SMS allowance, validity period, "
        "and eligibility. The exact activation code depends on the provider."
    ),
}
def ensure_kb():
    KB_DIR.mkdir(exist_ok=True)
    for n,c in DOCS.items():
        p=KB_DIR/n
        if not p.exists(): p.write_text(c,encoding="utf-8")
def retrieve(query, k=3):
    ensure_kb()

    docs = [
        (p.name, p.read_text(encoding="utf-8"))
        for p in KB_DIR.glob("*.txt")
    ]

    if not docs:
        return []

    texts = [x[1] for x in docs]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query])

    scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).ravel()

    ranked_indices = scores.argsort()[::-1]

    results = []

    for i in ranked_indices:
        if scores[i] <= 0.15:
            continue

        results.append({
            "source": docs[i][0],
            "score": float(scores[i]),
            "content": docs[i][1]
        })

        if len(results) >= k:
            break

    return results