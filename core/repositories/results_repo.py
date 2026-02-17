from pathlib import Path


class ResultsRepository:
    def __init__(self, logger):
        self.log = logger
        self.base = Path('results')
        self.base.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics: dict, name: str = 'metrics.json'):
        path = self.base/name
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.log.info(f"Saved metrics to {path}")

    def save_report(self, report, figs):
        # Minimální kompatibilní ukládání
        (self.base/"figs").mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figs or []):
            fig.savefig(self.base/"figs"/f"fig_{i+1}.png", dpi=120)
        # report lze uložit jako CSV/JSON podle dostupných atributů
        if hasattr(report, 'df'):
            report.df.to_csv(self.base/"report.csv", index=False)
