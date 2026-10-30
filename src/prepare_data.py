import gzip, json, glob, pandas as pd

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
rows = []
for f in glob.glob("data/raw/*.json.gz"):
    for v in json.load(gzip.open(f))["vulnerabilities"]:
        cve = v["cve"]
        m31 = cve.get("metrics", {}).get("cvssMetricV31", [])
        if not m31:
            continue
        # prefer NVD's own score (Primary), fall back to the CNA's (Secondary)
        m = next((x for x in m31 if x["type"] == "Primary"), m31[0])
        desc = next(d["value"] for d in cve["descriptions"] if d["lang"] == "en")
        if desc.startswith("** REJECT"):
            continue
        vec = m["cvssData"]["vectorString"]
        parts = dict(p.split(":") for p in vec.split("/")[1:])
        rows.append({"id": cve["id"], "published": cve["published"],
                     "description": desc, "vector": vec,
                     "score": m["cvssData"]["baseScore"],
                     "source": m["type"], **{k: parts[k] for k in METRICS}})

df = pd.DataFrame(rows).drop_duplicates("id")
df["published"] = pd.to_datetime(df["published"])
df[df.published < "2025-01-01"].to_parquet("data/processed/train.parquet")
df[df.published >= "2025-01-01"].to_parquet("data/processed/test.parquet")