# AI Hedge Fund

An AI-powered hedge fund that uses multiple agents to make trading decisions with **daily independent alpha generation backtesting**.

## 🚀 New Alpha-Focused Backtesting System

### Major System Redesign (v2.0)

The backtesting system has been completely redesigned from cumulative portfolio tracking to **daily independent alpha generation evaluation**:

#### Key Changes:

**1. Daily Independent Trading**
- Each trading day uses fixed capital allocation (default: $10,000)
- No cumulative position tracking - each day starts fresh
- No persistent portfolio state between days

**2. 30-Day Forward Alpha Measurement**
- Performance measured exactly 30 days forward from each trade date
- Each day's alpha calculated independently over its own 30-day window
- Alpha = Portfolio Return - SPY Benchmark Return

**3. SPY Benchmark Integration**
- SPY automatically fetched as benchmark (excluded from tradeable universe)
- Uses yfinance for reliable SPY data
- Alpha calculated against SPY performance for same period

**4. New Performance Metrics**
- **Win Rate**: Percentage of days with positive alpha
- **Average Alpha**: Mean alpha across all evaluated days  
- **Alpha Sharpe Ratio**: Risk-adjusted alpha performance
- **Alpha Distribution**: Statistical analysis of alpha generation

**5. Enhanced Visualization**
- Daily alpha time series plots
- Alpha distribution histograms  
- Cumulative alpha generation tracking
- Rolling win rate analysis

### Usage

#### Basic Alpha Backtesting
```bash
# Run with default settings (daily $10k capital allocation)
python -m src.backtester --tickers AAPL,MSFT,GOOGL --start-date 2024-01-01 --end-date 2024-02-01

# Custom daily capital allocation
python -m src.backtester --tickers AAPL,MSFT --daily-capital 25000 --start-date 2024-01-01

# Include all analysts
python -m src.backtester --tickers AAPL,MSFT,GOOGL --analysts-all --daily-capital 15000
```

#### Key Parameters:
- `--daily-capital`: Fixed capital used each day (default: $10,000)
- `--tickers`: Comma-separated list (SPY automatically excluded as benchmark)
- `--start-date` / `--end-date`: Backtest period
- `--analysts` / `--analysts-all`: Select trading analysts

### System Architecture

#### AlphaBacktester Class

**Core Methods:**
- `execute_daily_trades()`: Independent daily trade execution with fixed capital
- `calculate_trade_performance()`: 30-day forward performance evaluation  
- `update_alpha_statistics()`: Win/loss tracking and alpha accumulation
- `get_alpha_metrics()`: Comprehensive alpha-based performance metrics

**Data Flow:**
```
Day 1: Execute trades with $X → Store trade details
Day 2: Execute trades with $X → Store trade details  
...
Day 31: Evaluate Day 1 trades' 30-day performance → Calculate alpha → Update win/loss
Day 32: Evaluate Day 2 trades' 30-day performance → Calculate alpha → Update win/loss
```

#### Performance Metrics

**Primary Metrics:**
- **Win Rate**: `(win_days / total_evaluated_days) × 100`
- **Average Alpha**: `mean(all_alphas)`
- **Alpha Sharpe**: `mean(alphas) / std(alphas) × √252`

**Secondary Metrics:**
- Best/Worst Alpha Days
- Alpha Standard Deviation
- Total Evaluated Days
- Cumulative Alpha Generation

### Key Benefits

1. **Pure Alpha Focus**: Measures strategy's ability to generate alpha independent of market direction
2. **Consistent Evaluation**: Fixed capital ensures fair comparison across all trading days
3. **Forward-Looking**: 30-day evaluation window provides realistic performance assessment
4. **Benchmark-Relative**: All performance measured relative to SPY benchmark
5. **Statistical Rigor**: Win/loss tracking provides robust performance validation

### Backward Compatibility

The original `Backtester` class is maintained as an alias to `AlphaBacktester` for compatibility, but the new alpha-focused approach is recommended for all new backtests.

### Requirements

```bash
# Install dependencies including yfinance for SPY data
poetry install

# Or manually install yfinance
pip install yfinance>=0.2.28
```

### Example Output

```
Alpha-Focused Backtesting Configuration:
Trading Universe: AAPL, MSFT, GOOGL (SPY benchmark excluded)
Daily Capital: $10,000.00
Evaluation Method: 30-day forward alpha vs SPY

ALPHA PERFORMANCE SUMMARY:
Win Rate: 67.50%
Average Alpha: 0.85%
Alpha Sharpe: 1.23
Best Alpha Day: 4.21%
Worst Alpha Day: -2.15%
Total Evaluated Days: 40
Win Days: 27
Lose Days: 13
```

## Original Features

### Multi-Agent Architecture
- Warren Buffett Agent (Value Investing)
- Michael Burry Agent (Contrarian Deep Value) 
- Peter Lynch Agent (Growth at Reasonable Price)
- Cathie Wood Agent (Disruptive Innovation)
- And many more...

### LLM Integration
- OpenAI GPT models
- Anthropic Claude
- Local inference with Ollama
- Google Gemini support

### Real-time Data
- Financial datasets API integration
- Company fundamentals and metrics
- Insider trading data
- News sentiment analysis

### Portfolio Management
- Long/short position support
- Risk management integration
- Real-time decision making

## Getting Started

1. **Install Dependencies**
```bash
poetry install
```

2. **Set Environment Variables**
```bash
export FINANCIAL_DATASETS_API_KEY="your_api_key"
export OPENAI_API_KEY="your_openai_key"  # if using OpenAI
```

3. **Run Alpha Backtesting**
```bash
python -m src.backtester --tickers AAPL,MSFT --daily-capital 10000
```

4. **Run Live Trading** (Original System)
```bash
python -m src.main --tickers AAPL,MSFT,GOOGL
```

## Configuration

### API Keys
- `FINANCIAL_DATASETS_API_KEY`: For market data
- `OPENAI_API_KEY`: For GPT models
- `ANTHROPIC_API_KEY`: For Claude models
- `GROQ_API_KEY`: For Groq models

### Model Selection
Interactive model selection supports:
- OpenAI: GPT-4, GPT-3.5-turbo
- Anthropic: Claude-3.5 Sonnet, Claude-3 Haiku
- Local: Ollama models (llama3, mistral, etc.)

## Architecture

```
src/
├── agents/          # Trading strategy agents
├── backtester.py    # Alpha-focused backtesting engine  
├── tools/api.py     # Market data + SPY benchmark
├── utils/display.py # Alpha visualization
├── llm/models.py    # LLM integrations
└── main.py          # Live trading system
```

## License

MIT License - see LICENSE file for details.
