"""Die harten Limits müssen auf dem echten Ausführungspfad greifen, nicht nur im RiskManager."""
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
