import os
import yaml
import pandas as pd
from datetime import datetime

def run(config_path: str = "config.yaml"):
    # 1) load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("outputs", {}).get("folder", "out")
    os.makedirs(out_dir, exist_ok=True)

    # 2) TODO: replace with your real fetchers / model
    # For now, create a tiny dataframe so the pipeline completes.
    df = pd.DataFrame(
        [
            {"market": "example", "player": "Demo", "line": 50.5, "model_mean": 58.3, "edge_pct": 0.06},
        ]
    )

    # 3) write outputs
    csv_path = os.path.join(out_dir, "model_output.csv")
    xlsx_path = os.path.join(out_dir, cfg.get("outputs", {}).get("excel", "Slate_Model_Output.xlsx"))

    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Model", index=False)

    print(f"[{datetime.utcnow().isoformat()}Z] wrote: {csv_path} and {xlsx_path}")
