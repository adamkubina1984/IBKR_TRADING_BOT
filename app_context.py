from dataclasses import dataclass

from ibkr_trading_bot.core.repositories.data_repo import DataRepository
from ibkr_trading_bot.core.repositories.model_repo import ModelRepository
from ibkr_trading_bot.core.repositories.results_repo import ResultsRepository
from ibkr_trading_bot.core.services.data_download_service import DataDownloadService
from ibkr_trading_bot.core.services.evaluation_service import EvaluationService
from ibkr_trading_bot.core.services.live_bot_service import LiveBotService
from ibkr_trading_bot.core.services.model_training_service import ModelTrainingService
from ibkr_trading_bot.core.utils.logging_setup import setup_logging
from ibkr_trading_bot.core.utils.plotting import Plotting


@dataclass
class Services:
    data_download: DataDownloadService
    training: ModelTrainingService
    evaluation: EvaluationService
    live: LiveBotService
    models: ModelRepository

class AppContext:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(getattr(config, "log_level", "INFO"))
        # repositories
        self.data_repo = DataRepository(self.logger)
        self.model_repo = ModelRepository(self.logger)
        self.results_repo = ResultsRepository(self.logger)
        # utilities
        self.plotting = Plotting()
        # services
        self.services = Services(
            data_download=DataDownloadService(self.data_repo, self.logger),
            training=ModelTrainingService(self.model_repo, self.results_repo, self.logger),
            evaluation=EvaluationService(self.results_repo, self.plotting, self.logger),
            live=LiveBotService(self.model_repo, self.results_repo, self.logger),
            models=self.model_repo,
        )

    def data_frame_provider(self, symbol: str, timeframe: str):
        return self.data_repo.load_latest_data(symbol, timeframe)
