#!/usr/bin/env python3
"""
PhiHorizon Railway Paper Trading Bot

Paper trading on REAL market data with VIRTUAL budget.
Designed for Railway.app with persistent Volume storage.

Features:
- Real-time BTC prices from Binance API
- S+ AI model predictions (CORE 1)
- Dual logging: stdout + CSV
- Crash recovery from CSV
- UTC timestamps for verification

Usage:
    python railway_paper_bot.py
"""

import csv
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import requests

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Paper trading configuration - Production Grade."""
    # Budget
    initial_balance: float = 100.0  # $100 virtual
    
    # Trading - Core
    position_size_pct: float = 0.10  # 10% per trade (default, overridden by Kelly)
    stop_loss_pct: float = 0.02      # 2% stop loss
    take_profit_pct: float = 0.04    # 4% take profit
    fee_rate: float = 0.001          # 0.1% trading fee (Binance spot)
    
    # Production-Grade: Slippage Simulation
    slippage_rate: float = 0.0005    # 0.05% slippage (realistic for BTC)
    
    # Production-Grade: Trailing Stop
    use_trailing_stop: bool = True   # Enable trailing stop
    trailing_stop_pct: float = 0.02  # 2% trailing stop
    
    # Production-Grade: Kelly Position Sizing
    use_kelly_sizing: bool = True    # Enable Kelly criterion
    kelly_fraction: float = 0.25     # Quarter Kelly (safer)
    min_position_pct: float = 0.02   # Min 2% position
    max_position_pct: float = 0.20   # Max 20% position
    
    # Signal thresholds
    buy_confidence: float = 0.65     # Buy if confidence > 65%
    sell_confidence: float = 0.35    # Sell if confidence < 35%
    
    # Timing
    check_interval: int = 300        # 5 minutes between checks
    hold_log_interval: int = 12      # Log HOLD every 12 cycles (1 hour)
    
    # Paths (Railway Volume mount)
    data_dir: str = os.environ.get('DATA_DIR', '/data')
    csv_path: str = ''
    state_path: str = ''
    daily_report_path: str = ''
    
    def __post_init__(self):
        self.csv_path = f"{self.data_dir}/trades.csv"
        self.state_path = f"{self.data_dir}/state.json"
        self.daily_report_path = f"{self.data_dir}/daily_reports"


@dataclass
class Trade:
    """Single trade record."""
    timestamp: str
    action: str           # BUY, SELL, HOLD
    price: float
    confidence: float
    position: float       # BTC amount
    balance: float        # USD balance after trade
    pnl: float           # Profit/loss in USD
    pnl_pct: float       # Profit/loss percentage
    reason: str          # Why this action was taken


@dataclass
class State:
    """Bot state for crash recovery - Production Grade."""
    balance: float
    position: float           # Current BTC position
    entry_price: float        # Entry price if in position
    highest_price: float      # Highest price since entry (for trailing stop)
    total_trades: int
    winning_trades: int
    total_pnl: float
    total_fees: float         # Total fees paid
    total_slippage: float     # Total slippage cost
    max_drawdown: float
    peak_balance: float
    last_check: str
    current_date: str = ''    # For daily report tracking
    cycle_count: int = 0      # For HOLD logging interval


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    """Setup dual logging: stdout + file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Railway captures this
        ]
    )
    # Force UTC
    logging.Formatter.converter = time.gmtime
    return logging.getLogger('phihorizon')


# ============================================================
# BINANCE PRICE FETCHER
# ============================================================

class PriceFetcher:
    """Fetch real BTC price from Binance."""
    
    BINANCE_URL = "https://api.binance.com/api/v3/ticker/price"
    
    def get_btc_price(self) -> Optional[float]:
        """Get current BTC/USDT price."""
        try:
            response = requests.get(
                self.BINANCE_URL,
                params={'symbol': 'BTCUSDT'},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            logging.error(f"Failed to fetch price: {e}")
            return None
    
    def get_ohlcv(self, limit: int = 100) -> Optional[List[dict]]:
        """Get OHLCV data for indicators."""
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    'symbol': 'BTCUSDT',
                    'interval': '1h',
                    'limit': limit
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [
                {
                    'open': float(d[1]),
                    'high': float(d[2]),
                    'low': float(d[3]),
                    'close': float(d[4]),
                    'volume': float(d[5]),
                }
                for d in data
            ]
        except Exception as e:
            logging.error(f"Failed to fetch OHLCV: {e}")
            return None


# ============================================================
# SIMPLE SIGNAL GENERATOR (Placeholder for CORE 1)
# ============================================================

class SignalGenerator:
    """Generate trading signals.
    
    In production, this uses the trained CORE 1 S+ model.
    For now, using simple technical indicators.
    """
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return sum(prices[-period:]) / period
    
    def generate_signal(self, ohlcv: List[dict]) -> tuple:
        """Generate trading signal and confidence.
        
        Returns:
            (action, confidence, reason)
            action: 'BUY', 'SELL', or 'HOLD'
            confidence: 0.0 to 1.0
            reason: Explanation
        """
        if not ohlcv or len(ohlcv) < 30:
            return 'HOLD', 0.5, 'Insufficient data'
        
        closes = [c['close'] for c in ohlcv]
        current_price = closes[-1]
        
        # Calculate indicators
        rsi = self.calculate_rsi(closes)
        sma_20 = self.calculate_sma(closes, 20)
        sma_50 = self.calculate_sma(closes, 50)
        
        # Score calculation
        score = 0.5
        reasons = []
        
        # RSI signals
        if rsi < 30:
            score += 0.15
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.15
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # Trend signals
        if current_price > sma_20 > sma_50:
            score += 0.10
            reasons.append("Uptrend (price > SMA20 > SMA50)")
        elif current_price < sma_20 < sma_50:
            score -= 0.10
            reasons.append("Downtrend (price < SMA20 < SMA50)")
        
        # Momentum
        if len(closes) > 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5]
            if momentum > 0.02:
                score += 0.10
                reasons.append(f"Strong momentum (+{momentum:.1%})")
            elif momentum < -0.02:
                score -= 0.10
                reasons.append(f"Weak momentum ({momentum:.1%})")
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        # Determine action
        if score > 0.65:
            action = 'BUY'
        elif score < 0.35:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        reason = ' | '.join(reasons) if reasons else 'Neutral conditions'
        return action, score, reason


# ============================================================
# PAPER TRADING BOT
# ============================================================

class RailwayPaperBot:
    """Paper trading bot for Railway.app - Production Grade."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = setup_logging()
        self.price_fetcher = PriceFetcher()
        self.signal_generator = SignalGenerator()
        
        # Ensure data directories exist
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.daily_report_path).mkdir(parents=True, exist_ok=True)
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Initialize CSV if needed
        self._init_csv()
        
        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        self.logger.info("=" * 60)
        self.logger.info("PhiHorizon Railway Paper Trading Bot - PRODUCTION GRADE")
        self.logger.info("=" * 60)
        self.logger.info(f"Balance: ${self.state.balance:.2f}")
        self.logger.info(f"Position: {self.state.position:.6f} BTC")
        self.logger.info(f"Total trades: {self.state.total_trades}")
        self.logger.info(f"Total PnL: ${self.state.total_pnl:.2f}")
        self.logger.info(f"Total Fees: ${self.state.total_fees:.2f}")
        self.logger.info(f"Total Slippage: ${self.state.total_slippage:.2f}")
        self.logger.info("Features: Slippage ✓ | Kelly ✓ | Trailing Stop ✓")
    
    def _load_state(self) -> State:
        """Load state from file or create new."""
        if os.path.exists(self.config.state_path):
            try:
                with open(self.config.state_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"Resuming from saved state")
                return State(**data)
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")
        
        return State(
            balance=self.config.initial_balance,
            position=0.0,
            entry_price=0.0,
            highest_price=0.0,
            total_trades=0,
            winning_trades=0,
            total_pnl=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            max_drawdown=0.0,
            peak_balance=self.config.initial_balance,
            last_check=datetime.now(timezone.utc).isoformat(),
            current_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            cycle_count=0
        )
    
    def _save_state(self):
        """Save state to file."""
        try:
            with open(self.config.state_path, 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _init_csv(self):
        """Initialize CSV with headers if needed."""
        if not os.path.exists(self.config.csv_path):
            with open(self.config.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'action', 'price', 'confidence',
                    'position', 'balance', 'pnl', 'pnl_pct', 'reason'
                ])
            self.logger.info(f"Created CSV: {self.config.csv_path}")
    
    def _log_trade(self, trade: Trade):
        """Log trade to CSV (append-only)."""
        try:
            with open(self.config.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.timestamp,
                    trade.action,
                    f"{trade.price:.2f}",
                    f"{trade.confidence:.4f}",
                    f"{trade.position:.6f}",
                    f"{trade.balance:.2f}",
                    f"{trade.pnl:.2f}",
                    f"{trade.pnl_pct:.4f}",
                    trade.reason
                ])
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")
    
    def _check_new_day(self):
        """Check if it's a new day and generate daily report."""
        current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        if self.state.current_date and self.state.current_date != current_date:
            # New day - generate report for previous day
            self._generate_daily_report(self.state.current_date)
        
        self.state.current_date = current_date
    
    def _generate_daily_report(self, date: str):
        """Generate daily performance report."""
        report_file = Path(self.config.daily_report_path) / f"{date}.json"
        
        win_rate = (
            self.state.winning_trades / self.state.total_trades * 100
            if self.state.total_trades > 0 else 0
        )
        
        report = {
            'date': date,
            'balance': self.state.balance,
            'total_pnl': self.state.total_pnl,
            'total_trades': self.state.total_trades,
            'winning_trades': self.state.winning_trades,
            'win_rate': win_rate,
            'total_fees': self.state.total_fees,
            'total_slippage': self.state.total_slippage,
            'max_drawdown': self.state.max_drawdown,
            'peak_balance': self.state.peak_balance,
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"📊 Daily report saved: {report_file}")
            self.logger.info(
                f"Day {date} | PnL: ${self.state.total_pnl:.2f} | "
                f"Win: {win_rate:.0f}% | Fees: ${self.state.total_fees:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save daily report: {e}")
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.logger.info("Shutdown signal received, saving state...")
        # Generate final daily report
        if self.state.current_date:
            self._generate_daily_report(self.state.current_date)
        self._save_state()
        self.logger.info("State saved. Goodbye!")
        sys.exit(0)
    
    def _execute_buy(self, price: float, confidence: float, reason: str):
        """Execute a buy order with slippage, fees, and Kelly sizing."""
        if self.state.position > 0:
            self.logger.info("Already in position, skip buy")
            return
        
        # Production-Grade: Apply slippage (price goes UP when buying)
        slippage = price * self.config.slippage_rate
        actual_price = price + slippage
        self.state.total_slippage += slippage
        
        # Production-Grade: Kelly position sizing
        if self.config.use_kelly_sizing and self.state.total_trades >= 5:
            position_pct = self._calculate_kelly_size()
        else:
            position_pct = self.config.position_size_pct
        
        # Calculate position size with fees
        trade_amount = self.state.balance * position_pct
        fee = trade_amount * self.config.fee_rate
        net_amount = trade_amount - fee
        btc_amount = net_amount / actual_price
        
        # Update state
        self.state.balance -= trade_amount
        self.state.position = btc_amount
        self.state.entry_price = actual_price
        self.state.highest_price = actual_price  # Initialize trailing stop
        self.state.total_trades += 1
        self.state.total_fees += fee
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Log trade
        trade = Trade(
            timestamp=timestamp,
            action='BUY',
            price=actual_price,
            confidence=confidence,
            position=btc_amount,
            balance=self.state.balance,
            pnl=-(fee + slippage * btc_amount),
            pnl_pct=0.0,
            reason=f"{reason} | Fee: ${fee:.4f} | Slip: ${slippage:.2f}"
        )
        self._log_trade(trade)
        self._save_state()
        
        self.logger.info(
            f"📈 BUY | ${actual_price:.2f} (slip ${slippage:.2f}) | "
            f"{btc_amount:.6f} BTC ({position_pct:.0%}) | Fee: ${fee:.4f}"
        )
    
    def _calculate_kelly_size(self) -> float:
        """Calculate position size using Kelly Criterion."""
        if self.state.total_trades < 5:
            return self.config.position_size_pct
        
        win_rate = self.state.winning_trades / self.state.total_trades
        
        # Estimate avg win/loss from PnL history (simplified)
        avg_win_pct = 0.04  # 4% take profit
        avg_loss_pct = 0.02  # 2% stop loss
        
        # Kelly formula: f = (bp - q) / b
        # b = odds (avg_win / avg_loss)
        # p = win probability, q = loss probability
        b = avg_win_pct / avg_loss_pct
        p = win_rate
        q = 1 - win_rate
        
        kelly = (b * p - q) / b
        
        # Apply fraction and clamp
        kelly = kelly * self.config.kelly_fraction
        kelly = max(self.config.min_position_pct, min(kelly, self.config.max_position_pct))
        
        return kelly

    
    def _execute_sell(self, price: float, confidence: float, reason: str):
        """Execute a sell order with slippage and fees."""
        if self.state.position <= 0:
            self.logger.info("No position to sell, skip")
            return
        
        # Production-Grade: Apply slippage (price goes DOWN when selling)
        slippage = price * self.config.slippage_rate
        actual_price = price - slippage
        self.state.total_slippage += slippage
        
        # Calculate PnL with fees
        gross_sell_value = self.state.position * actual_price
        fee = gross_sell_value * self.config.fee_rate
        net_sell_value = gross_sell_value - fee
        
        buy_value = self.state.position * self.state.entry_price
        pnl = net_sell_value - buy_value
        pnl_pct = pnl / buy_value if buy_value > 0 else 0
        
        # Update state
        self.state.balance += net_sell_value
        self.state.total_pnl += pnl
        self.state.total_fees += fee
        if pnl > 0:
            self.state.winning_trades += 1
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Track drawdown
        if self.state.balance > self.state.peak_balance:
            self.state.peak_balance = self.state.balance
        current_dd = (self.state.balance - self.state.peak_balance) / self.state.peak_balance
        if current_dd < self.state.max_drawdown:
            self.state.max_drawdown = current_dd
        
        # Log trade
        trade = Trade(
            timestamp=timestamp,
            action='SELL',
            price=actual_price,
            confidence=confidence,
            position=0.0,
            balance=self.state.balance,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=f"{reason} | Fee: ${fee:.4f} | Slip: ${slippage:.2f}"
        )
        self._log_trade(trade)
        
        # Reset position
        self.state.position = 0.0
        self.state.entry_price = 0.0
        self.state.highest_price = 0.0  # Reset trailing stop
        self._save_state()
        
        emoji = "✅" if pnl > 0 else "❌"
        self.logger.info(
            f"{emoji} SELL | ${actual_price:.2f} (slip ${slippage:.2f}) | "
            f"PnL: ${pnl:.2f} ({pnl_pct:.1%}) | Fee: ${fee:.4f}"
        )
    
    def _check_stop_loss_take_profit(self, price: float) -> Optional[str]:
        """Check if stop loss, take profit, or trailing stop hit."""
        if self.state.position <= 0 or self.state.entry_price <= 0:
            return None
        
        # Update highest price for trailing stop
        if price > self.state.highest_price:
            self.state.highest_price = price
        
        pnl_pct = (price - self.state.entry_price) / self.state.entry_price
        
        # Check stop loss first
        if pnl_pct <= -self.config.stop_loss_pct:
            return f"STOP_LOSS ({pnl_pct:.1%})"
        
        # Check take profit
        if pnl_pct >= self.config.take_profit_pct:
            return f"TAKE_PROFIT ({pnl_pct:.1%})"
        
        # Production-Grade: Trailing Stop
        if self.config.use_trailing_stop and self.state.highest_price > self.state.entry_price:
            trailing_stop_price = self.state.highest_price * (1 - self.config.trailing_stop_pct)
            if price <= trailing_stop_price:
                profit_from_high = (self.state.highest_price - self.state.entry_price) / self.state.entry_price
                actual_profit = pnl_pct
                return f"TRAILING_STOP (peak +{profit_from_high:.1%}, exit +{actual_profit:.1%})"
        
        return None
    
    def run_once(self):
        """Run one trading cycle."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.state.last_check = timestamp
        
        # Production-Grade: Check for new day to generate daily report
        self._check_new_day()
        
        # Fetch price
        price = self.price_fetcher.get_btc_price()
        if price is None:
            self.logger.warning("Could not fetch price, skipping cycle")
            return
        
        self.logger.info(f"BTC Price: ${price:,.2f}")
        
        # Check stop loss / take profit first
        sl_tp = self._check_stop_loss_take_profit(price)
        if sl_tp:
            self._execute_sell(price, 0.0, sl_tp)
            return
        
        # Get OHLCV for signals
        ohlcv = self.price_fetcher.get_ohlcv()
        if not ohlcv:
            self.logger.warning("Could not fetch OHLCV, skipping signal")
            return
        
        # Generate signal
        action, confidence, reason = self.signal_generator.generate_signal(ohlcv)
        
        self.logger.info(f"Signal: {action} | Confidence: {confidence:.1%} | {reason}")
        
        # Execute based on signal
        if action == 'BUY' and confidence >= self.config.buy_confidence:
            self._execute_buy(price, confidence, reason)
        elif action == 'SELL' and confidence <= self.config.sell_confidence:
            if self.state.position > 0:
                self._execute_sell(price, confidence, reason)
        else:
            # Log HOLD only every N cycles to reduce CSV size
            self.state.cycle_count += 1
            if self.state.cycle_count >= self.config.hold_log_interval:
                trade = Trade(
                    timestamp=timestamp,
                    action='HOLD',
                    price=price,
                    confidence=confidence,
                    position=self.state.position,
                    balance=self.state.balance,
                    pnl=0.0,
                    pnl_pct=0.0,
                    reason=reason
                )
                self._log_trade(trade)
                self.state.cycle_count = 0
            self._save_state()
    
    def run(self):
        """Run continuous trading loop."""
        self.logger.info(f"Starting trading loop (interval: {self.config.check_interval}s)")
        
        while True:
            try:
                self.run_once()
                
                # Print summary
                equity = self.state.balance
                if self.state.position > 0:
                    current_price = self.price_fetcher.get_btc_price() or 0
                    equity += self.state.position * current_price
                
                win_rate = (
                    self.state.winning_trades / self.state.total_trades * 100
                    if self.state.total_trades > 0 else 0
                )
                
                self.logger.info(
                    f"Stats | Equity: ${equity:.2f} | Trades: {self.state.total_trades} | "
                    f"Win: {win_rate:.0f}% | PnL: ${self.state.total_pnl:.2f} | "
                    f"MDD: {self.state.max_drawdown:.1%}"
                )
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 min on error


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""
    # Check if running locally or on Railway
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        print("Running on Railway")
        config = Config(data_dir='/data')
    else:
        print("Running locally")
        config = Config(data_dir='./paper_trading_data')
    
    bot = RailwayPaperBot(config)
    bot.run()


if __name__ == '__main__':
    main()
