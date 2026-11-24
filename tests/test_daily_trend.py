"""Tagestrend: Regeln ohne Blick in die Zukunft, Abgleich-Schritt im Bot."""
import numpy as np
import pandas as pd
import pytest

from strategies.daily_trend import DailyTrendStrategy, TrendVote, donchian_hold, ema_direction, vote_count
from tests.conftest import run


def _frame(close):
    close = pd.Series(close, index=pd.date_range('2024-01-01', periods=len(close)), dtype=float)
    return pd.DataFrame({'high': close * 1.01, 'low': close * 0.99, 'close': close})


def test_donchian_einstieg_und_ausstieg():
    # 25 Tage flach, Ausbruch, dann Absturz unter das 10-Tage-Tief
    df = _frame([100] * 25 + [110, 112, 114] + [80])
    hold = donchian_hold(df.high, df.low, df.close, 20, 10)
    assert not hold.iloc[24]
    assert hold.iloc[25] and hold.iloc[27]
    assert not hold.iloc[28]


def test_donchian_kanal_schliesst_die_aktuelle_kerze_aus():
    """Ein neues Hoch allein reicht nicht: der Schluss muss über dem Hoch der VORIGEN Tage liegen."""
    df = _frame([100] * 30)
    df.iloc[-1, df.columns.get_loc('high')] = 150.0   # nur Docht, Schluss bleibt 100
    assert not donchian_hold(df.high, df.low, df.close, 20, 10).iloc[-1]


@pytest.mark.parametrize('rule', ['donchian', 'vote'])
def test_kein_blick_in_die_zukunft(rule):
    """Der Zustand an Tag t darf sich nicht ändern, wenn spätere Kerzen dazukommen."""
    rng = np.random.default_rng(7)
    df = _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400))))
    full = donchian_hold(df.high, df.low, df.close, 20, 10) if rule == 'donchian' else vote_count(df.close) >= 3
    for t in (250, 300, 399):
        part = df.iloc[:t + 1]
        cut = donchian_hold(part.high, part.low, part.close, 20, 10) if rule == 'donchian' else vote_count(part.close) >= 3
        assert bool(cut.iloc[-1]) == bool(full.iloc[t])


def test_evaluate_braucht_genug_historie():
    strat = DailyTrendStrategy({'rule': 'donchian'})
    assert strat.evaluate('BTC_EUR', _frame([100] * 60)) is None
    vote = strat.evaluate('BTC_EUR', _frame([100] * 70 + [120]))
    assert vote.hold and vote.upper == pytest.approx(120 * 1.01)


def _vote(symbol, hold):
    return TrendVote(symbol=symbol, votes=int(hold), hold=hold, candle_date=pd.Timestamp('2026-01-01').to_pydatetime(),
                     close=1.0, rules={'ausbruch': hold}, upper=1.0, lower=1.0)


def _bot_with_votes(make_bot, votes, fail=()):
    bot = make_bot()

    async def fetch_daily(symbol):
        if symbol in fail:
            raise RuntimeError('Kraken nicht erreichbar')
        return symbol

    bot.daily_trend.fetch_daily = fetch_daily
    bot.daily_trend.evaluate = lambda symbol, daily: _vote(symbol, votes[symbol])
    return bot


def test_schritt_kauft_und_verkauft(make_bot):
    votes = {'BTC_EUR': True}
    bot = _bot_with_votes(make_bot, votes)
    run(bot._daily_trend_step('BTC_EUR', {}))
    assert bot.portfolio.positions['BTC_EUR'].market_type == 'daily_trend'
    votes['BTC_EUR'] = False
    run(bot._daily_trend_step('BTC_EUR', {}))
    assert bot.portfolio.positions == {}


def test_schritt_ohne_frischen_preis_handelt_nicht(make_bot):
    bot = _bot_with_votes(make_bot, {'BTC_EUR': True})
    bot.crypto_feed.prices.pop('BTC_EUR')
    run(bot._daily_trend_step('BTC_EUR', {}))
    assert bot.portfolio.positions == {}


def test_schritt_kauft_nicht_nach_tageslimit(make_bot):
    bot = _bot_with_votes(make_bot, {'BTC_EUR': True})
    bot.portfolio.daily_start_balance = bot.portfolio.equity / 0.85
    run(bot._daily_trend_step('BTC_EUR', {}))
    assert bot.portfolio.positions == {}


def test_fehler_bei_einer_coin_stoppt_die_anderen_nicht(make_bot):
    bot = _bot_with_votes(make_bot, {'BTC_EUR': True, 'ETH_EUR': True, 'SOL_EUR': True})

    def evaluate(symbol, daily):
        if symbol == 'BTC_EUR':
            raise ValueError('kaputte Kerze')
        return _vote(symbol, True)

    bot.daily_trend.evaluate = evaluate
    bot.running = True

    async def one_round():
        import asyncio
        task = asyncio.create_task(bot._daily_trend_loop())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(bot.portfolio.positions) == 2:
                break
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    run(one_round())
    assert set(bot.portfolio.positions) == {'ETH_EUR', 'SOL_EUR'}


# --- EMA 20/50 long/short (Swing-1) ---

def test_ema_richtung_ohne_blick_in_die_zukunft():
    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))))
    full = ema_direction(close, 20, 50)
    assert set(full.unique()) <= {1, -1}
    for t in (200, 400, 599):
        assert ema_direction(close.iloc[:t + 1], 20, 50).iloc[-1] == full.iloc[t]


def test_ema_evaluate_long_und_short():
    strat = DailyTrendStrategy({'rule': 'ema'})
    assert strat.timeframe == '4h'
    up = _frame(np.linspace(100, 200, 200))
    down = _frame(np.linspace(200, 100, 200))
    assert strat.evaluate('BTC_EUR', up).target == 1
    vote = strat.evaluate('BTC_EUR', down)
    assert vote.target == -1 and not vote.hold
    assert 'short' in vote.describe()
    assert strat.evaluate('BTC_EUR', up.iloc[:150]) is None   # EMA noch nicht eingeschwungen


def test_offene_kerze_wird_verworfen():
    strat = DailyTrendStrategy({'rule': 'ema', 'timeframe': '4h'})
    t0 = pd.Timestamp('2026-11-24 00:00')
    rows = [[int((t0 + pd.Timedelta(hours=4 * i)).timestamp() * 1000), 1, 1, 1, 1, 1] for i in range(3)]
    # 09:00: 00:00 und 04:00 sind zu, die 08:00-Kerze läuft noch
    df = strat.closed_candles(rows, now=pd.Timestamp('2026-11-24 09:00', tz='UTC'))
    assert list(df.index) == [t0, t0 + pd.Timedelta(hours=4)]
    day = DailyTrendStrategy({'rule': 'donchian'})
    rows = [[int((t0 + pd.Timedelta(days=i)).timestamp() * 1000), 1, 1, 1, 1, 1] for i in range(2)]
    assert len(day.closed_candles(rows, now=pd.Timestamp('2026-11-25 23:00', tz='UTC'))) == 1


def _dir_vote(symbol, direction):
    return TrendVote(symbol=symbol, votes=direction, hold=direction > 0,
                     candle_date=pd.Timestamp('2026-01-01').to_pydatetime(), close=1.0,
                     rules={'ema': direction > 0}, direction=direction, ema_fast=1.0, ema_slow=1.0)


def _ema_bot(make_bot, dirs, spot_only=False):
    bot = make_bot({'general': {'start_capital': 1000},
                    'exchange_rules': {'spot_only': spot_only},
                    'strategies': {'daily_trend': {'rule': 'ema'}},
                    'risk': {'max_concurrent_positions': 4}})

    async def fetch_daily(symbol):
        return symbol

    bot.daily_trend.fetch_daily = fetch_daily
    bot.daily_trend.evaluate = lambda symbol, daily: _dir_vote(symbol, dirs[symbol])
    return bot


def test_ema_wechselt_von_long_auf_short(make_bot):
    dirs = {'BTC_EUR': 1}
    bot = _ema_bot(make_bot, dirs)
    run(bot._daily_trend_step('BTC_EUR', {}))
    assert bot.portfolio.positions['BTC_EUR'].side == 'long'
    run(bot._daily_trend_step('BTC_EUR', {}))           # gleiche Richtung: nichts tun
    assert len(bot.portfolio.trades) == 0
    dirs['BTC_EUR'] = -1
    run(bot._daily_trend_step('BTC_EUR', {}))
    pos = bot.portfolio.positions['BTC_EUR']
    assert pos.side == 'short' and pos.leverage == 1 and pos.size == pytest.approx(200, rel=0.01)
    assert len(bot.portfolio.trades) == 1


def test_short_gewinnt_wenn_der_kurs_faellt(make_bot):
    bot = _ema_bot(make_bot, {'BTC_EUR': -1})
    run(bot._daily_trend_step('BTC_EUR', {}))
    trade = bot.portfolio.close_position('BTC_EUR', exit_price=45000.0, fees=0.0, strategy='daily_trend')
    assert trade.side == 'short' and trade.pnl > 15   # 200 € · 10 % minus Einstiegsgebühr


def test_spot_bot_shortet_nie(make_bot):
    """Panda-9 und Trend-1 bleiben spot_only: ein Short-Soll heißt dort draußen bleiben."""
    bot = _ema_bot(make_bot, {'BTC_EUR': -1}, spot_only=True)
    run(bot._daily_trend_step('BTC_EUR', {}))
    assert bot.portfolio.positions == {}
