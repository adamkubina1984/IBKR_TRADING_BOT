from ibkr_trading_bot.core.datasource import tradingview_client as tv_client_module


class _DummySettings:
    _store: dict[str, str] = {}

    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None):
        return self._store.get(key, default)

    def setValue(self, key, value):
        self._store[key] = value

    def remove(self, key):
        self._store.pop(key, None)

    def sync(self):
        return None


def test_saved_tv_credentials_roundtrip(monkeypatch):
    _DummySettings._store = {}
    monkeypatch.setattr(tv_client_module, "QSettings", _DummySettings)
    monkeypatch.delenv("TV_USERNAME", raising=False)
    monkeypatch.delenv("TV_PASSWORD", raising=False)

    tv_client_module.save_tv_credentials("alice", "secret")
    username, password = tv_client_module.load_saved_tv_credentials()
    resolved_user, resolved_pwd, source = tv_client_module.resolve_tv_credentials()

    assert username == "alice"
    assert password == "secret"
    assert resolved_user == "alice"
    assert resolved_pwd == "secret"
    assert source == "env"


def test_env_credentials_override_saved(monkeypatch):
    _DummySettings._store = {}
    monkeypatch.setattr(tv_client_module, "QSettings", _DummySettings)
    tv_client_module.save_tv_credentials("saved_user", "saved_pwd")
    monkeypatch.setenv("TV_USERNAME", "env_user")
    monkeypatch.setenv("TV_PASSWORD", "env_pwd")

    resolved_user, resolved_pwd, source = tv_client_module.resolve_tv_credentials()

    assert resolved_user == "env_user"
    assert resolved_pwd == "env_pwd"
    assert source == "env"
