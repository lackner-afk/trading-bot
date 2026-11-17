"""Tagestrend: Regeln ohne Blick in die Zukunft, Abgleich-Schritt im Bot."""
import numpy as np
import pandas as pd
import pytest

from strategies.daily_trend import DailyTrendStrategy, TrendVote, donchian_hold, vote_count
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
