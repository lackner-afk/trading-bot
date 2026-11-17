"""Die harten Limits müssen auf dem echten Ausführungspfad greifen, nicht nur im RiskManager."""
import pytest

from tests.conftest import run, long_signal


def test_einstieg_ohne_drawdown_wird_ausgefuehrt(make_bot):
    bot = make_bot()
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert 'BTC_EUR' in bot.portfolio.positions


def test_tageslimit_blockiert_neue_einstiege(make_bot):
    """check_trade meldet CLOSE_ALL — das darf nicht wie ALLOW behandelt werden."""
    bot = make_bot()
    # 12 % Tagesverlust simulieren (Limit 10 %)
    bot.portfolio.daily_start_balance = bot.portfolio.equity / 0.88
    assert bot.portfolio.get_daily_drawdown() >= bot.risk_manager.max_daily_drawdown
    run(bot._execute_signal(long_signal(), strategy_name='momentum'))
    assert bot.portfolio.positions == {}


def test_pause_blockiert_einstiege(make_bot):
    bot = make_bot()
    bot.trading_paused = True
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert bot.portfolio.positions == {}


def test_spot_only_verwirft_shorts_und_hebel(make_bot):
    from strategies.crypto_scalper import SignalType
    bot = make_bot()
    sig = long_signal()
    sig.signal_type = SignalType.SHORT
    run(bot._execute_signal(sig, strategy_name='momentum'))
    assert bot.portfolio.positions == {}
    sig = long_signal(symbol='ETH_EUR', price=2000.0)
    sig.suggested_leverage = 20
    run(bot._execute_signal(sig, strategy_name='momentum'))
    assert bot.portfolio.positions['ETH_EUR'].leverage == 1


def test_max_positionen_aus_config(make_bot):
    bot = make_bot()
    for sym, px in [('BTC_EUR', 50000.0), ('ETH_EUR', 2000.0), ('SOL_EUR', 100.0)]:
        run(bot._execute_signal(long_signal(symbol=sym, price=px), strategy_name='daily_trend'))
    assert len(bot.portfolio.positions) == 2


# --- Risk-Check mit echter Ordergröße, Mindestorder als ausdrückliche Ausnahme ---

def test_mindestorder_ausnahme_erlaubt_30_prozent(make_bot):
    bot = make_bot()   # settings.yaml: allow_min_order_exception: true, 100 €
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert bot.portfolio.positions['BTC_EUR'].size == 30.0


def test_ohne_ausnahme_handelt_ein_100_euro_konto_nicht(make_bot):
    bot = make_bot({'exchange_rules': {'allow_min_order_exception': False}})
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert bot.portfolio.positions == {}


def test_ausnahme_nie_ueber_halbes_kapital(make_bot):
    bot = make_bot({'general': {'start_capital': 55}})   # 30 € wären 55 %
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert bot.portfolio.positions == {}


def test_grosses_konto_bleibt_bei_20_prozent(make_bot):
    bot = make_bot({'general': {'start_capital': 1000}})
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert bot.portfolio.positions['BTC_EUR'].size == 200.0


def test_kuerzung_bei_drawdown_wird_befolgt(make_bot):
    bot = make_bot({'general': {'start_capital': 1000}})
    bot.portfolio.daily_start_balance = bot.portfolio.equity / 0.95   # 5 % Tagesverlust
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    # 200 € · (1 − 5 %/10 %) = 100 €
    assert bot.portfolio.positions['BTC_EUR'].size == pytest.approx(100.0)


def test_kuerzung_unter_mindestorder_heisst_kein_trade(make_bot):
    bot = make_bot()   # 100 €: 30 € lassen sich nicht kürzen
    bot.portfolio.daily_start_balance = bot.portfolio.equity / 0.95
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert bot.portfolio.positions == {}


def test_risiko_je_trade_gilt_auch_fuer_die_mindestorder(make_bot):
    """Die Ausnahme betrifft nur die Positionsgrenze: 30 € mit 10 % Stop = 3 € > 2 €."""
    bot = make_bot()
    sig = long_signal(stop_loss=45000.0)
    run(bot._execute_signal(sig, strategy_name='confluence'))
    assert bot.portfolio.positions == {}
