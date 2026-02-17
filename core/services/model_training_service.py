

class ModelTrainingService:
    def __init__(self, model_repo, results_repo, logger):
        self.models = model_repo
        self.results = results_repo
        self.log = logger

    def rolling_retrain(self, df, features_cfg_path: str, **kwargs):
        # Zachování logiky: zde jen placeholder – původní implementace zůstává v projektu,
        # GUI může nadále volat existující funkce; tohle API je připraveno pro přesun bez změny signatur.
        self.log.info("rolling_retrain placeholder: zachováváme původní trénovací tok")
        best_model, metrics = None, {}
        return best_model, metrics


from ibkr_trading_bot.model.train_models import train_and_evaluate_model as _train_eval


class ModelTrainingService(ModelTrainingService):  # extend existing class
    def train_and_evaluate_model(self, X_train, y_train, model_name: str, param_grid=None, on_progress=None):
        """Tenký proxy wrapper nad původní funkcí – beze změny chování."""
        return _train_eval(X_train, y_train, model_name, param_grid=param_grid, on_progress=on_progress)
