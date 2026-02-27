"""On-chain whale activity tracker — detects large moves before price reacts.

Sources:
1. Exchange trades API (via ccxt) — detect large individual trades
2. Blockchain.info API (free, no auth) — BTC mempool & network activity
3. Volume anomaly detection — unusual volume spikes signal whale accumulation

Returns a whale_score from -1.0 (bearish whales dumping) to +1.0 (bullish whales buying).
"""

import time

import httpx

# Per-symbol cache: {symbol: {"data": ..., "time": ...}}
_whale_cache: dict[str, dict] = {}
_WHALE_CACHE_TTL = 600  # 10 minutes


def fetch_whale_activity(exchange, symbol: str) -> dict:
    """Detect whale activity through multiple signals.

    Returns:
        whale_score: -1.0 (bearish) to +1.0 (bullish)
        large_buys: count of large buy trades detected
        large_sells: count of large sell trades detected
        volume_anomaly: current vol / avg vol ratio (>2.0 = spike)
        details: list of human-readable detail strings
    """
    now = time.time()
    cached = _whale_cache.get(symbol)
    if cached and (now - cached["time"]) < _WHALE_CACHE_TTL:
        return cached["data"]

    result = {
        "whale_score": 0.0,
        "large_buys": 0,
        "large_sells": 0,
        "volume_anomaly": 1.0,
        "details": [],
    }

    signals = []  # (name, score, weight)

    # ── Signal 1: Large trade detection via exchange recent trades ──
    try:
        trades = exchange.fetch_trades(symbol, limit=200)
        if trades:
            amounts_usd = [t["amount"] * t["price"] for t in trades]
            median_size = sorted(amounts_usd)[len(amounts_usd) // 2]
            # Whale threshold: 10x median trade size or $10k minimum
            threshold = max(median_size * 10, 10_000)

            large_buys = [t for t in trades
                          if t["amount"] * t["price"] >= threshold and t["side"] == "buy"]
            large_sells = [t for t in trades
                           if t["amount"] * t["price"] >= threshold and t["side"] == "sell"]

            result["large_buys"] = len(large_buys)
            result["large_sells"] = len(large_sells)

            total_large = len(large_buys) + len(large_sells)
            if total_large > 0:
                buy_ratio = len(large_buys) / total_large
                # Also weight by USD volume of large trades
                buy_vol = sum(t["amount"] * t["price"] for t in large_buys)
                sell_vol = sum(t["amount"] * t["price"] for t in large_sells)
                total_vol = buy_vol + sell_vol
                vol_ratio = (buy_vol / total_vol - 0.5) * 2 if total_vol > 0 else 0

                signal = (buy_ratio - 0.5) * 2 * 0.5 + vol_ratio * 0.5
                signals.append(("large_trades", signal, 0.45))

                if buy_vol > sell_vol * 1.5:
                    result["details"].append(
                        f"Whale accumulation: {len(large_buys)} large buys "
                        f"(${buy_vol:,.0f}) vs {len(large_sells)} sells (${sell_vol:,.0f})"
                    )
                elif sell_vol > buy_vol * 1.5:
                    result["details"].append(
                        f"Whale distribution: {len(large_sells)} large sells "
                        f"(${sell_vol:,.0f}) vs {len(large_buys)} buys (${buy_vol:,.0f})"
                    )
                else:
                    result["details"].append(
                        f"Whale activity mixed: {len(large_buys)} buys, {len(large_sells)} sells"
                    )

            # Volume anomaly: total recent trade volume vs what we'd expect
            if len(amounts_usd) > 50:
                recent_vol = sum(amounts_usd[-50:])
                older_vol = sum(amounts_usd[:50])
                if older_vol > 0:
                    anomaly = recent_vol / older_vol
                    result["volume_anomaly"] = round(anomaly, 2)
                    if anomaly > 2.0:
                        signals.append(("vol_spike", 0.3, 0.15))
                        result["details"].append(
                            f"Volume spike: {anomaly:.1f}x recent vs older trades"
                        )
                    elif anomaly < 0.5:
                        signals.append(("vol_drop", -0.1, 0.10))
                        result["details"].append(
                            f"Volume drop: {anomaly:.1f}x — low interest"
                        )
    except Exception:
        pass

    # ── Signal 2: BTC mempool activity (blockchain.info, free no auth) ──
    if "BTC" in symbol.upper():
        try:
            resp = httpx.get(
                "https://blockchain.info/q/unconfirmedcount", timeout=5
            )
            if resp.status_code == 200:
                unconfirmed = int(resp.text.strip())
                # Normal: 5k-20k, High (panic/large moves): >30k, Low: <3k
                if unconfirmed > 40000:
                    signals.append(("mempool", -0.3, 0.15))
                    result["details"].append(
                        f"BTC mempool congested: {unconfirmed:,} unconfirmed — potential panic"
                    )
                elif unconfirmed > 25000:
                    signals.append(("mempool", -0.1, 0.10))
                    result["details"].append(
                        f"BTC mempool elevated: {unconfirmed:,} unconfirmed"
                    )
                elif unconfirmed < 3000:
                    signals.append(("mempool", 0.1, 0.05))
                    result["details"].append(
                        f"BTC mempool calm: {unconfirmed:,} unconfirmed"
                    )
        except Exception:
            pass

    # ── Signal 3: Hashrate proxy (blockchain.info) ──
    if "BTC" in symbol.upper():
        try:
            resp = httpx.get(
                "https://blockchain.info/q/hashrate", timeout=5
            )
            if resp.status_code == 200:
                # High hashrate = miners confident = bullish
                # We just note it as context, small weight
                hashrate_gh = float(resp.text.strip())
                result["details"].append(
                    f"BTC hashrate: {hashrate_gh / 1e6:.0f} EH/s"
                )
        except Exception:
            pass

    # ── Combine signals ──
    if signals:
        total_weight = sum(w for _, _, w in signals)
        weighted_score = sum(s * w for _, s, w in signals) / total_weight if total_weight > 0 else 0
        result["whale_score"] = round(max(-1.0, min(1.0, weighted_score)), 3)

    _whale_cache[symbol] = {"data": result, "time": now}
    return result
