#!/usr/bin/env python3
"""
PhiHorizon V5.5 Paper Trading Bot

Paper trading system with $100 virtual funds.
Supports:
1. OKX Demo Trading API (with real-time prices)
2. Local simulation mode (offline)

Usage:
    python paper_trading_bot.py --mode local
    python paper_trading_bot.py --mode okx  # Requires API keys
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import sentiment loader for Filter D
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.sentiment_loader import SentimentLoader
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False

# Import sentiment model for Filter G (V6.1 LSTM)
try:
    from src.models.sentiment_model import SentimentModel
    HAS_SENTIMENT_MODEL = True
except ImportError:
    HAS_SENTIMENT_MODEL = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# V5.5 RESEARCH-BACKED CONFIGURATION
# ============================================================

@dataclass
class V55Config:
    """V5.5 optimized configuration with A+B filters."""
    # Account
    initial_balance: float = 100.0  # $100 virtual
    
    # Target & Holding (from research)
    target_return: float = 0.01     # 1.0%
    holding_period: int = 24        # 24 hours
    
    # Costs
    transaction_fee: float = 0.001  # 0.1% per side
    slippage: float = 0.0005        # 0.05%
    
    # Risk Management
    kelly_fraction: float = 0.25    # Quarter Kelly
    max_position_pct: float = 0.10  # Max 10%
    min_position_pct: float = 0.01  # Min 1%
    
    # Stops
    atr_stop_mult: float = 2.5      # 2.5x ATR
    atr_tp_mult: float = 4.0        # 4x ATR
    
    # Probability filter
    prob_threshold: float = 0.55    # [V6.1.2] Lowered from 0.60 for more trades
    
    # ===== FILTER A: TREND =====
    # [V6.1.2] Changed from MA200 to MA100 - faster, better for crypto
    use_trend_filter: bool = True   # Enable trend filter
    trend_fast_period: int = 50     # Fast MA period
    trend_slow_period: int = 100    # [V6.1.2] Was 200, now 100 for less strict
    
    # ===== FILTER B: VOLATILITY =====
    use_volatility_filter: bool = True  # Enable volatility filter
    min_atr_pct: float = 0.01       # Min ATR as % of price (1%)
    max_atr_pct: float = 0.05       # Max ATR as % of price (5%)
    
    # ===== FILTER D: FEAR/GREED =====
    use_fear_greed_filter: bool = True  # Enable Fear/Greed filter
    extreme_fear_threshold: int = 20    # Below = Extreme Fear (Bullish)
    extreme_greed_threshold: int = 80   # Above = Extreme Greed (Block)
    
    # ===== FILTER E: FUNDING RATE =====
    # [V6.1.2] Adjusted threshold to 0.03% for more realistic filtering
    use_funding_filter: bool = True     # Enable Funding Rate filter
    funding_positive_threshold: float = 0.0003  # [V6.1.2] 0.03% = overleveraged
    funding_negative_threshold: float = -0.0003 # <-0.03% = overleveraged short
    
    # ===== FILTER F: WHALE/ON-CHAIN =====
    # [V6.1.2] Adjusted to -3% for more realistic filtering
    use_whale_filter: bool = True       # Enable Whale activity filter
    whale_accumulation_threshold: float = 0.03  # >3% outflow = accumulation
    whale_distribution_threshold: float = -0.03 # <-3% inflow = distribution
    
    # ===== FILTER G: SENTIMENT LSTM (CONFIRMATION MODE) =====
    # [V6.1.2] Changed from FILTER to CONFIRMATION - scales position, never blocks
    use_sentiment_filter: bool = True   # Enable Sentiment (as confirmation)
    sentiment_bullish_scale: float = 1.2   # Scale up 20% on bullish
    sentiment_bearish_scale: float = 0.8   # Scale down 20% on bearish


@dataclass
class Trade:
    """Record of a single trade."""
    trade_id: str
    timestamp: str
    direction: str  # 'long' or 'short'
    entry_price: float
    position_size: float
    position_dollars: float
    stop_loss: float
    take_profit: float
    atr: float
    probability: float
    status: str = 'open'  # 'open', 'closed'
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class Portfolio:
    """Paper trading portfolio state."""
    balance: float = 100.0
    equity: float = 100.0
    open_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    
    # Stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 100.0
    
    def to_dict(self) -> dict:
        return {
            'balance': self.balance,
            'equity': self.equity,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_pnl': self.total_pnl,
            'total_return': (self.equity - 100) / 100,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.open_trades),
        }


class PaperTradingBot:
    """
    Paper trading bot implementing V5.5 strategy.
    
    Uses virtual funds to simulate trading without real risk.
    """
    
    def __init__(self, config: V55Config = None, mode: str = 'local'):
        self.config = config or V55Config()
        self.mode = mode
        self.portfolio = Portfolio(balance=self.config.initial_balance)
        self.portfolio.equity = self.config.initial_balance
        self.portfolio.peak_equity = self.config.initial_balance
        
        # State file for persistence
        self.state_file = Path('paper_trading_state.json')
        
        # Price history for indicators
        self.price_history: List[dict] = []
        
        # Filter G: Sentiment Model (V6.1 LSTM)
        self.sentiment_model = None
        if self.config.use_sentiment_filter and HAS_SENTIMENT_MODEL:
            try:
                self.sentiment_model = SentimentModel(
                    project_root=str(Path(__file__).parent.parent)
                )
                if self.sentiment_model.is_loaded:
                    logger.info("Filter G: Sentiment LSTM loaded")
                else:
                    logger.warning("Filter G: Model not loaded (TF unavailable)")
            except Exception as e:
                logger.warning(f"Filter G: Failed to load - {e}")
        
        logger.info(f"Paper Trading Bot initialized")
        logger.info(f"Mode: {mode}")
        logger.info(f"Initial Balance: ${self.config.initial_balance}")
    
    def calculate_indicators(self, prices: pd.DataFrame) -> dict:
        """Calculate trading indicators."""
        if len(prices) < 50:
            return {}
        
        close = prices['close']
        high = prices['high']
        low = prices['low']
        
        # Returns
        returns_24h = close.pct_change(24).iloc[-1]
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # SMA
        sma_24 = close.rolling(24).mean().iloc[-1]
        sma_168 = close.rolling(168).mean().iloc[-1]
        
        return {
            'returns_24h': returns_24h,
            'atr': atr,
            'rsi': rsi,
            'sma_24': sma_24,
            'sma_168': sma_168,
            'close': close.iloc[-1],
        }
    
    def calculate_signal_probability(self, indicators: dict) -> float:
        """
        Calculate signal probability (0-1).
        
        This is a simplified version. In production, use the trained model.
        """
        if not indicators:
            return 0.5
        
        score = 0.5
        
        # Momentum (returns_24h)
        ret = indicators.get('returns_24h', 0)
        if ret > 0.02:
            score += 0.1
        elif ret > 0.01:
            score += 0.05
        elif ret < -0.02:
            score -= 0.1
        
        # RSI
        rsi = indicators.get('rsi', 50)
        if 40 < rsi < 60:  # Neutral zone
            score += 0.05
        elif rsi < 30:  # Oversold
            score += 0.1
        elif rsi > 70:  # Overbought
            score -= 0.05
        
        # Trend
        sma_24 = indicators.get('sma_24', 0)
        sma_168 = indicators.get('sma_168', 0)
        if sma_24 > sma_168:  # Uptrend
            score += 0.1
        else:
            score -= 0.05
        
        return max(0, min(1, score))
    
    def calculate_position_size(self) -> float:
        """Calculate position size using Kelly criterion."""
        # Get historical performance
        if len(self.portfolio.closed_trades) >= 10:
            wins = [t for t in self.portfolio.closed_trades[-50:] if t.pnl > 0]
            losses = [t for t in self.portfolio.closed_trades[-50:] if t.pnl <= 0]
            
            if wins and losses:
                win_rate = len(wins) / (len(wins) + len(losses))
                avg_win = np.mean([t.pnl_pct for t in wins])
                avg_loss = abs(np.mean([t.pnl_pct for t in losses]))
                
                # Kelly formula
                b = avg_win / avg_loss if avg_loss > 0 else 1
                kelly = (win_rate * b - (1 - win_rate)) / b
                kelly = kelly * self.config.kelly_fraction
                
                return max(self.config.min_position_pct,
                          min(kelly, self.config.max_position_pct))
        
        # Default to minimum
        return self.config.min_position_pct
    
    def open_trade(self, current_price: float, atr: float, probability: float) -> Optional[Trade]:
        """Open a new trade."""
        if probability < self.config.prob_threshold:
            logger.info(f"Signal filtered: prob={probability:.2%} < {self.config.prob_threshold:.0%}")
            return None
        
        # Calculate position
        position_pct = self.calculate_position_size()
        position_dollars = self.portfolio.balance * position_pct
        position_size = position_dollars / current_price
        
        # Calculate stops
        stop_distance = atr * self.config.atr_stop_mult
        tp_distance = atr * self.config.atr_tp_mult
        
        stop_loss = current_price - stop_distance
        take_profit = current_price + tp_distance
        
        # Create trade
        trade = Trade(
            trade_id=f"T{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            direction='long',
            entry_price=current_price,
            position_size=position_size,
            position_dollars=position_dollars,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            probability=probability,
        )
        
        self.portfolio.open_trades.append(trade)
        self.portfolio.total_trades += 1
        
        logger.info(f"📈 TRADE OPENED: {trade.trade_id}")
        logger.info(f"   Entry: ${current_price:.2f}")
        logger.info(f"   Size: ${position_dollars:.2f} ({position_pct:.1%})")
        logger.info(f"   Stop: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
        
        return trade
    
    def check_and_close_trades(self, current_price: float):
        """Check open trades for stop/TP hit."""
        for trade in self.portfolio.open_trades[:]:
            should_close = False
            exit_reason = None
            exit_price = current_price
            
            if trade.direction == 'long':
                if current_price <= trade.stop_loss:
                    should_close = True
                    exit_reason = 'STOP_LOSS'
                    exit_price = trade.stop_loss  # Assume exact fill
                elif current_price >= trade.take_profit:
                    should_close = True
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = trade.take_profit
            
            if should_close:
                self.close_trade(trade, exit_price, exit_reason)
    
    def close_trade(self, trade: Trade, exit_price: float, reason: str):
        """Close a trade and update portfolio."""
        # Calculate PnL (including costs)
        gross_pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        costs = self.config.transaction_fee * 2 + self.config.slippage * 2
        net_pnl_pct = gross_pnl_pct - costs
        net_pnl = trade.position_dollars * net_pnl_pct
        
        # Update trade
        trade.status = 'closed'
        trade.exit_price = exit_price
        trade.exit_time = datetime.now().isoformat()
        trade.pnl = net_pnl
        trade.pnl_pct = net_pnl_pct
        trade.exit_reason = reason
        
        # Update portfolio
        self.portfolio.balance += net_pnl
        self.portfolio.equity = self.portfolio.balance
        self.portfolio.total_pnl += net_pnl
        
        if net_pnl > 0:
            self.portfolio.winning_trades += 1
        else:
            self.portfolio.losing_trades += 1
        
        # Update drawdown
        if self.portfolio.equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = self.portfolio.equity
        
        current_dd = (self.portfolio.equity - self.portfolio.peak_equity) / self.portfolio.peak_equity
        if current_dd < self.portfolio.max_drawdown:
            self.portfolio.max_drawdown = current_dd
        
        # Move to closed trades
        self.portfolio.open_trades.remove(trade)
        self.portfolio.closed_trades.append(trade)
        
        emoji = "✅" if net_pnl > 0 else "❌"
        logger.info(f"{emoji} TRADE CLOSED: {trade.trade_id}")
        logger.info(f"   Exit: ${exit_price:.2f} | Reason: {reason}")
        logger.info(f"   PnL: ${net_pnl:.2f} ({net_pnl_pct:.2%})")
        logger.info(f"   Balance: ${self.portfolio.balance:.2f}")
    
    def save_state(self):
        """Save portfolio state to file."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': self.portfolio.to_dict(),
            'closed_trades': [
                {
                    'id': t.trade_id,
                    'entry': t.entry_price,
                    'exit': t.exit_price,
                    'pnl': t.pnl,
                    'reason': t.exit_reason,
                }
                for t in self.portfolio.closed_trades[-20:]
            ],
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {self.state_file}")
    
    def load_state(self):
        """Load portfolio state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.portfolio.balance = state['portfolio']['balance']
            self.portfolio.equity = state['portfolio']['equity']
            self.portfolio.total_trades = state['portfolio']['total_trades']
            self.portfolio.winning_trades = state['portfolio']['winning_trades']
            self.portfolio.losing_trades = state['portfolio']['losing_trades']
            self.portfolio.total_pnl = state['portfolio']['total_pnl']
            self.portfolio.max_drawdown = state['portfolio']['max_drawdown']
            
            logger.info(f"State loaded: Balance=${self.portfolio.balance:.2f}")
    
    def print_summary(self):
        """Print trading summary."""
        stats = self.portfolio.to_dict()
        
        print("\n" + "="*50)
        print("📊 PAPER TRADING SUMMARY")
        print("="*50)
        print(f"Balance:      ${stats['balance']:.2f}")
        print(f"Total PnL:    ${stats['total_pnl']:.2f} ({stats['total_return']:.2%})")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate:     {stats['win_rate']:.1%}")
        print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
        print(f"Open Positions: {stats['open_positions']}")
        print("="*50)
    
    def run_simulation(self, prices: pd.DataFrame, verbose: bool = True):
        """
        Run paper trading simulation on historical prices.
        
        Args:
            prices: DataFrame with OHLCV data
            verbose: Print each trade
        """
        logger.info(f"Starting simulation with {len(prices)} candles...")
        
        for i in range(50, len(prices) - 24):  # Need 50 for indicators, 24 for holding
            current = prices.iloc[i]
            current_price = current['close']
            
            # Calculate indicators
            history = prices.iloc[max(0, i-200):i+1]
            indicators = self.calculate_indicators(history)
            
            if not indicators:
                continue
            
            # Check existing trades
            self.check_and_close_trades(current_price)
            
            # Generate signal
            probability = self.calculate_signal_probability(indicators)
            
            # Open new trade if no open positions
            if len(self.portfolio.open_trades) == 0:
                # Check probability filter
                if probability <= self.config.prob_threshold:
                    continue
                
                # ===== FILTER A: TREND FILTER =====
                if self.config.use_trend_filter:
                    sma_24 = indicators.get('sma_24', 0)
                    sma_168 = indicators.get('sma_168', 0)
                    if sma_24 <= sma_168:  # Not in uptrend
                        continue  # Skip this signal
                
                # ===== FILTER B: VOLATILITY FILTER =====
                if self.config.use_volatility_filter:
                    atr = indicators.get('atr', 0)
                    atr_pct = atr / current_price if current_price > 0 else 0
                    
                    # Skip if volatility too low (choppy) or too high (risky)
                    if atr_pct < self.config.min_atr_pct:
                        continue
                    if atr_pct > self.config.max_atr_pct:
                        continue
                
                # ===== V6.0 NEW: FILTER D (FEAR/GREED) =====
                # Note: For simulation, we use current day's F&G
                # In live, this would be fetched real-time
                if self.config.use_fear_greed_filter and hasattr(self, 'current_fear_greed'):
                    fg_value = self.current_fear_greed
                    if fg_value is not None:
                        # Block trade during Extreme Greed (high risk)
                        if fg_value >= self.config.extreme_greed_threshold:
                            continue  # Skip - market too greedy
                
                # All filters passed - open trade
                self.open_trade(
                    current_price=current_price,
                    atr=indicators['atr'],
                    probability=probability
                )
        
        # Close any remaining at end
        for trade in self.portfolio.open_trades[:]:
            self.close_trade(trade, prices.iloc[-1]['close'], 'END_OF_DATA')
        
        self.print_summary()
        self.save_state()


def main():
    parser = argparse.ArgumentParser(description='V5.5 Paper Trading Bot')
    parser.add_argument('--mode', choices=['local', 'okx'], default='local',
                       help='Trading mode: local simulation or OKX demo')
    parser.add_argument('--balance', type=float, default=100.0,
                       help='Initial balance (default: $100)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from saved state')
    args = parser.parse_args()
    
    # Config
    config = V55Config(initial_balance=args.balance)
    
    # Create bot
    bot = PaperTradingBot(config=config, mode=args.mode)
    
    if args.resume:
        bot.load_state()
    
    if args.mode == 'local':
        # Load sample data
        try:
            import yfinance as yf
            print("Loading BTC data...")
            btc = yf.download('BTC-USD', period='3mo', interval='1h', progress=False)
            btc = btc.reset_index()
            
            # Handle MultiIndex columns from yfinance
            if isinstance(btc.columns, pd.MultiIndex):
                btc.columns = [col[0].lower() if col[0] else col[1].lower() for col in btc.columns]
            else:
                btc.columns = [str(c).lower() for c in btc.columns]
            
            # Rename common variants
            btc = btc.rename(columns={'adj close': 'close', 'adj_close': 'close'})
            
            print(f"Loaded {len(btc)} candles")
            print(f"Columns: {list(btc.columns)}")
            bot.run_simulation(btc)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Please install yfinance: pip install yfinance")
    
    else:  # OKX mode
        print("\n" + "="*50)
        print("OKX DEMO TRADING MODE")
        print("="*50)
        print("To use OKX demo trading:")
        print("1. Create demo API keys at OKX")
        print("2. Set environment variables:")
        print("   OKX_API_KEY, OKX_SECRET, OKX_PASSPHRASE")
        print("3. Run this script with --mode okx")
        print("="*50)


if __name__ == '__main__':
    main()
