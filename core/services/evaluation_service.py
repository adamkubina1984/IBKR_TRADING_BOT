class EvaluationService:
    def __init__(self, results_repo, plotting, logger):
        self.results = results_repo
        self.plot = plotting
        self.log = logger

    def backtest_and_report(self, df, model, **kwargs):
        # Placeholder API – výpočty zůstávají v utils/metrics|model_io.
        class Report:
            pass

        report = Report()
        report.equity = kwargs.get('equity')
        report.trade_returns = kwargs.get('trade_returns')
        figs = [self.plot.equity_curve(report), self.plot.payout_histogram(report)]
        self.results.save_report(report, figs)
        return report


from ibkr_trading_bot.utils.metrics import calculate_metrics as _calculate_metrics


class EvaluationService(EvaluationService):
    def calculate_metrics(self, *args, **kwargs):
        return _calculate_metrics(*args, **kwargs)
