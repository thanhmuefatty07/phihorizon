# Title: Built a "Conscious" Trading Bot using Integrated Information Theory (Phi) + Walk-Forward Optimization. Here are the results.

**Subreddit:** r/algotrading
**Flair:** Project / Strategy

---

Hi everyone,

I’ve been working on an experimental trading system concepts from **Integrated Information Theory (IIT)** – predominantly used in neuroscience to measure consciousness – to financial time series.

The core idea is to calculate **Φ (Phi)** for the market. I treat the market as a system where:
1. **Price Entropy** = Complexity
2. **Volume-Price Correlation** = Coherence
3. **Multi-scale Trend Alignment** = Integration

When all three align (High Phi), the market is "conscious/efficient" and trends are sustainable. When Phi is low, the market is "unconscious/noisy" (random walk).

### The Stack
- **Engine:** Python (Custom backtester, no framework overhead)
- **Models:** XGBoost (Regime) + LSTM (Sequence). Experimenting with **Mamba** layers.
- **Validation:** Rigorous **Walk-Forward Optimization** with Bayesian tuning. No static params.
- **Risk:** Volatility-adjusted sizing + Regime-based stops.

### Preliminary Results (Walk-Forward)
I'm currently running heavy benchmarks on Kaggle/Colab, but initial 5-window walk-forward tests on BTC/USDT (5m) show:
- **Sharpe:** ~1.2 - 1.5 (Out-of-sample)
- **Win Rate:** ~56%
- **Correlation to Buy & Hold:** Low (< 0.3)

### The "Aha!" Moment
The "Consciousness" metric (Phi) actually seems to predict **breakout failures**. High Phi breakouts tend to stick. Low Phi breakouts tend to fake out. It filters noise better than ADX in my tests.

### Request for Feedback
1. Has anyone else applied Information Theory / Entropy metrics to regime filters?
2. What’s your experience with Mamba/SSMs vs Transformers for OHLCV data?
3. I'm considering open-sourcing the "Consciousness Metrics" module. Would this be useful?

Happy to discuss the math/code logic!

*(Disclaimer: Not selling anything yet, just sharing R&D)*
