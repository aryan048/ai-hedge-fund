import sys

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools

from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.analysts import ANALYST_ORDER
from src.main import run_hedge_fund
from src.tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
    get_spy_data,
    calculate_spy_return,
)
from src.utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable
from src.utils.ollama import ensure_ollama_and_model

init(autoreset=True)


class AlphaBacktester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        daily_capital: float = 10000,
        model_name: str = "gpt-4.1",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
    ):
        """
        Alpha-focused backtester that evaluates daily independent trading decisions.
        
        :param agent: The trading agent (Callable).
        :param tickers: List of tickers to backtest (SPY excluded as it's the benchmark).
        :param start_date: Start date string (YYYY-MM-DD).
        :param end_date: End date string (YYYY-MM-DD).
        :param daily_capital: Fixed capital amount used for each day's trades.
        :param model_name: Which LLM model name to use (gpt-4, etc).
        :param model_provider: Which LLM provider (OpenAI, etc).
        :param selected_analysts: List of analyst names or IDs to incorporate.
        """
        self.agent = agent
        # Ensure SPY is not in the tradeable universe
        self.tickers = [t for t in tickers if t.upper() != 'SPY']
        self.start_date = start_date
        self.end_date = end_date
        self.daily_capital = daily_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts

        # Daily trade tracking for 30-day forward evaluation
        self.daily_trades = {}  # {trade_date: {ticker: {action, quantity, price}}}
        self.trade_evaluations = {}  # {trade_date: {alpha, portfolio_return, spy_return}}
        
        # Performance tracking
        self.win_days = 0
        self.lose_days = 0
        self.total_alpha = 0.0
        self.alpha_history = []
        self.spy_data = None

    def execute_daily_trades(self, decisions: dict, current_prices: dict, current_date: str) -> dict:
        """
        Execute independent daily trades with fixed capital allocation.
        Returns trade details for future evaluation.
        """
        trade_details = {}
        
        for ticker in self.tickers:
            decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
            action = decision.get("action", "hold").lower()
            quantity = decision.get("quantity", 0)
            
            if action in ["buy", "sell", "short", "cover"] and quantity > 0:
                price = current_prices[ticker]
                
                # For simplicity, limit trade size to available daily capital
                max_quantity = int(self.daily_capital / price) if price > 0 else 0
                executed_quantity = min(int(quantity), max_quantity)
                
                if executed_quantity > 0:
                    trade_details[ticker] = {
                        "action": action,
                        "quantity": executed_quantity,
                        "price": price,
                        "cost": executed_quantity * price
                    }
            
        return trade_details

    def calculate_trade_performance(self, trade_date: str, evaluation_date: str) -> dict:
        """
        Calculate the performance of trades executed on trade_date, evaluated on evaluation_date.
        Returns portfolio return, SPY return, and alpha.
        """
        if trade_date not in self.daily_trades:
            return {"portfolio_return": 0.0, "spy_return": 0.0, "alpha": 0.0}
        
        trade_details = self.daily_trades[trade_date]
        total_portfolio_return = 0.0
        total_invested = 0.0
        
        for ticker, trade in trade_details.items():
            try:
                # Get price data for the evaluation period
                price_data = get_price_data(ticker, trade_date, evaluation_date)
                if price_data.empty:
                    continue
                
                entry_price = trade["price"]
                exit_price = price_data.iloc[-1]["close"]
                quantity = trade["quantity"]
                action = trade["action"]
                
                # Calculate return based on action
                if action == "buy":
                    trade_return = (exit_price - entry_price) / entry_price
                elif action == "sell":
                    # Assume we had the position and sold it
                    trade_return = 0.0  # Simplified for now
                elif action == "short":
                    trade_return = (entry_price - exit_price) / entry_price
                else:  # cover
                    trade_return = 0.0  # Simplified for now
                
                position_value = quantity * entry_price
                total_portfolio_return += trade_return * position_value
                total_invested += position_value
                
            except Exception as e:
                print(f"Error evaluating trade for {ticker} on {trade_date}: {e}")
                continue
        
        # Calculate weighted portfolio return
        portfolio_return = total_portfolio_return / total_invested if total_invested > 0 else 0.0
        
        # Calculate SPY return for the same period
        spy_return = 0.0
        if self.spy_data is not None:
            spy_return = calculate_spy_return(self.spy_data, trade_date, evaluation_date)
        
        # Calculate alpha
        alpha = portfolio_return - spy_return
        
        return {
            "portfolio_return": portfolio_return,
            "spy_return": spy_return,
            "alpha": alpha
        }

    def update_alpha_statistics(self, alpha: float):
        """Update running alpha statistics."""
        self.alpha_history.append(alpha)
        self.total_alpha += alpha
        
        if alpha > 0:
            self.win_days += 1
        else:
            self.lose_days += 1

    def get_alpha_metrics(self) -> dict:
        """Calculate comprehensive alpha-based performance metrics."""
        if not self.alpha_history:
            return {}
        
        alphas = np.array(self.alpha_history)
        total_evaluated_days = len(alphas)
        
        return {
            "win_rate": (self.win_days / total_evaluated_days * 100) if total_evaluated_days > 0 else 0,
            "average_alpha": np.mean(alphas),
            "alpha_std": np.std(alphas),
            "alpha_sharpe": np.mean(alphas) / np.std(alphas) * np.sqrt(252) if np.std(alphas) > 0 else 0,
            "best_alpha": np.max(alphas),
            "worst_alpha": np.min(alphas),
            "total_evaluated_days": total_evaluated_days,
            "win_days": self.win_days,
            "lose_days": self.lose_days
        }

    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period."""
        print("\nPre-fetching data for the entire backtest period...")

        # Convert end_date string to datetime, fetch up to 1 year before
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")
        
        # Extend end date for 30-day forward evaluation
        extended_end_date = (end_date_dt + timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch SPY benchmark data
        print("Fetching SPY benchmark data...")
        self.spy_data = get_spy_data(start_date_str, extended_end_date)
        if self.spy_data.empty:
            print("Warning: Could not fetch SPY data. Alpha calculations may be inaccurate.")

        for ticker in self.tickers:
            # Fetch price data for the entire period, plus buffer for forward evaluation
            get_prices(ticker, start_date_str, extended_end_date)

            # Fetch financial metrics
            get_financial_metrics(ticker, self.end_date, limit=10)

            # Fetch insider trades
            get_insider_trades(ticker, self.end_date, start_date=self.start_date, limit=1000)

            # Fetch company news
            get_company_news(ticker, self.end_date, start_date=self.start_date, limit=1000)

        print("Data pre-fetch complete.")

    def run_backtest(self):
        """Run the alpha-focused backtest with daily independent evaluation."""
        # Pre-fetch all data at the start
        self.prefetch_data()

        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []

        print("\nStarting alpha-focused backtest...")
        print(f"Daily capital allocation: ${self.daily_capital:,.2f}")
        print(f"Forward evaluation period: 30 days")

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

            # Skip if there's no prior day to look back
            if lookback_start == current_date_str:
                continue

            # Get current prices for all tickers
            try:
                current_prices = {}
                missing_data = False

                for ticker in self.tickers:
                    try:
                        price_data = get_price_data(ticker, previous_date_str, current_date_str)
                        if price_data.empty:
                            print(f"Warning: No price data for {ticker} on {current_date_str}")
                            missing_data = True
                            break
                        current_prices[ticker] = price_data.iloc[-1]["close"]
                    except Exception as e:
                        print(f"Error fetching price for {ticker} on {current_date_str}: {e}")
                        missing_data = True
                        break

                if missing_data:
                    print(f"Skipping trading day {current_date_str} due to missing price data")
                    continue

            except Exception as e:
                print(f"Error fetching prices for {current_date_str}: {e}")
                continue

            # Execute the agent's trading decisions
            output = self.agent(
                tickers=self.tickers,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio={"cash": self.daily_capital, "positions": {}},  # Fresh capital each day
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
            )
            decisions = output["decisions"]
            analyst_signals = output["analyst_signals"]

            # Execute daily independent trades
            trade_details = self.execute_daily_trades(decisions, current_prices, current_date_str)
            self.daily_trades[current_date_str] = trade_details

            # Calculate 30-day forward performance for today's trades immediately
            evaluation_end_date = (current_date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            if trade_details:  # Only if we actually made trades
                try:
                    performance = self.calculate_trade_performance(current_date_str, evaluation_end_date)
                    self.trade_evaluations[current_date_str] = performance
                    self.update_alpha_statistics(performance["alpha"])
                    
                    # Print evaluation results
                    print(f"\nAlpha evaluation for {current_date_str} (30-day forward):")
                    print(f"Portfolio Return: {performance['portfolio_return']:.2%}")
                    print(f"SPY Return: {performance['spy_return']:.2%}")
                    print(f"Alpha: {performance['alpha']:.2%}")
                except Exception as e:
                    print(f"Warning: Could not calculate alpha for {current_date_str}: {e}")
                    # Still add a placeholder to maintain data consistency
                    self.trade_evaluations[current_date_str] = {"portfolio_return": 0.0, "spy_return": 0.0, "alpha": 0.0}

            # Build table rows for display
            date_rows = []
            for ticker in self.tickers:
                ticker_signals = {}
                for agent_name, signals in analyst_signals.items():
                    if ticker in signals:
                        ticker_signals[agent_name] = signals[ticker]

                bullish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish"])
                bearish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish"])
                neutral_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral"])

                # Get trade details for this ticker and date
                trade = trade_details.get(ticker, {})
                action = trade.get("action", decisions.get(ticker, {}).get("action", "hold"))
                quantity = trade.get("quantity", 0)

                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        price=current_prices[ticker],
                        shares_owned=0,  # No persistent positions in alpha mode
                        position_value=trade.get("cost", 0),
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )

            # Add alpha summary row
            current_metrics = self.get_alpha_metrics()
            date_rows.append(
                format_alpha_summary_row(
                    date=current_date_str,
                    win_rate=current_metrics.get("win_rate", 0),
                    average_alpha=current_metrics.get("average_alpha", 0),
                    alpha_sharpe=current_metrics.get("alpha_sharpe", 0),
                    total_evaluated_days=current_metrics.get("total_evaluated_days", 0),
                    has_data=bool(current_metrics)
                )
            )

            table_rows.extend(date_rows)
            print_backtest_results(table_rows)

        # Store final performance metrics
        self.performance_metrics = self.get_alpha_metrics()
        return self.performance_metrics



    def analyze_performance(self):
        """Analyze and display alpha-focused performance results."""
        if not self.alpha_history:
            print("No alpha data found. Please run the backtest first.")
            return pd.DataFrame()

        print(f"\n{Fore.WHITE}{Style.BRIGHT}ALPHA GENERATION PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        
        metrics = self.get_alpha_metrics()
        
        print(f"Win Rate: {Fore.GREEN if metrics['win_rate'] >= 50 else Fore.RED}{metrics['win_rate']:.2f}%{Style.RESET_ALL}")
        print(f"Average Alpha: {Fore.GREEN if metrics['average_alpha'] >= 0 else Fore.RED}{metrics['average_alpha']:.2%}{Style.RESET_ALL}")
        print(f"Alpha Sharpe Ratio: {Fore.YELLOW}{metrics['alpha_sharpe']:.2f}{Style.RESET_ALL}")
        print(f"Best Alpha Day: {Fore.GREEN}{metrics['best_alpha']:.2%}{Style.RESET_ALL}")
        print(f"Worst Alpha Day: {Fore.RED}{metrics['worst_alpha']:.2%}{Style.RESET_ALL}")
        print(f"Total Evaluated Days: {Fore.WHITE}{metrics['total_evaluated_days']}{Style.RESET_ALL}")
        print(f"Win Days: {Fore.GREEN}{metrics['win_days']}{Style.RESET_ALL}")
        print(f"Lose Days: {Fore.RED}{metrics['lose_days']}{Style.RESET_ALL}")

        # Plot alpha distribution over time
        plt.figure(figsize=(15, 10))
        
        # Alpha over time
        plt.subplot(2, 2, 1)
        plt.plot(self.alpha_history, color="blue", alpha=0.7)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.title("Daily Alpha Over Time")
        plt.ylabel("Alpha")
        plt.xlabel("Trading Day")
        plt.grid(True, alpha=0.3)
        
        # Alpha histogram
        plt.subplot(2, 2, 2)
        plt.hist(self.alpha_history, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
        plt.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        plt.title("Alpha Distribution")
        plt.xlabel("Alpha")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        # Cumulative alpha
        plt.subplot(2, 2, 3)
        cumulative_alpha = np.cumsum(self.alpha_history)
        plt.plot(cumulative_alpha, color="green", linewidth=2)
        plt.title("Cumulative Alpha Generation")
        plt.ylabel("Cumulative Alpha")
        plt.xlabel("Trading Day")
        plt.grid(True, alpha=0.3)
        
        # Rolling win rate
        plt.subplot(2, 2, 4)
        if len(self.alpha_history) >= 20:
            rolling_wins = pd.Series(self.alpha_history).rolling(20).apply(lambda x: (x > 0).sum() / len(x) * 100)
            plt.plot(rolling_wins, color="orange", linewidth=2)
            plt.axhline(y=50, color="red", linestyle="--", alpha=0.7)
            plt.title("20-Day Rolling Win Rate")
            plt.ylabel("Win Rate (%)")
            plt.xlabel("Trading Day")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        # Return alpha history as DataFrame
        alpha_df = pd.DataFrame({
            'Alpha': self.alpha_history,
            'Cumulative_Alpha': np.cumsum(self.alpha_history),
            'Win': np.array(self.alpha_history) > 0
        })
        
        return alpha_df


def format_alpha_summary_row(
    date: str,
    win_rate: float,
    average_alpha: float,
    alpha_sharpe: float,
    total_evaluated_days: int,
    has_data: bool = True,
    days_until_alpha: int = 0,
) -> list:
    """Format a summary row for alpha-focused results."""
    if not has_data or total_evaluated_days == 0:
        # Show initial status when no alpha data is available yet
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}ALPHA SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Shares
            "",  # Position Value
            f"{Fore.YELLOW}No trades yet{Style.RESET_ALL}",  # No data message
            f"{Fore.YELLOW}Awaiting trades{Style.RESET_ALL}",  # Status
            f"{Fore.YELLOW}--{Style.RESET_ALL}",  # Placeholder
        ]
    else:
        # Show actual alpha metrics
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}ALPHA SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Shares
            "",  # Position Value
            f"{Fore.GREEN if win_rate >= 50 else Fore.RED}{win_rate:.1f}%{Style.RESET_ALL}",  # Win Rate
            f"{Fore.GREEN if average_alpha >= 0 else Fore.RED}{average_alpha:.2%}{Style.RESET_ALL}",  # Avg Alpha
            f"{Fore.YELLOW}{alpha_sharpe:.2f}{Style.RESET_ALL}",  # Alpha Sharpe
        ]


# Backward compatibility - keep old class name as alias
Backtester = AlphaBacktester


### 4. Run the Backtest #####
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run alpha-focused backtesting simulation")
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL). SPY excluded as benchmark.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--daily-capital",
        type=float,
        default=10000,
        help="Daily capital allocation amount (default: 10000)",
    )
    parser.add_argument(
        "--analysts",
        type=str,
        required=False,
        help="Comma-separated list of analysts to use (e.g., michael_burry,other_analyst)",
    )
    parser.add_argument(
        "--analysts-all",
        action="store_true",
        help="Use all available analysts (overrides --analysts)",
    )
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")

    args = parser.parse_args()

    # Parse tickers from comma-separated string, exclude SPY
    tickers = []
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip().upper() != 'SPY']
        if not tickers:
            print("Warning: SPY excluded from tradeable universe (used as benchmark only)")

    # Parse analysts from command-line flags
    selected_analysts = None
    if args.analysts_all:
        selected_analysts = [a[1] for a in ANALYST_ORDER]
    elif args.analysts:
        selected_analysts = [a.strip() for a in args.analysts.split(",") if a.strip()]
    else:
        # Choose analysts interactively
        choices = questionary.checkbox(
            "Use the Space bar to select/unselect analysts.",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
            validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
            style=questionary.Style(
                [
                    ("checkbox-selected", "fg:green"),
                    ("selected", "fg:green noinherit"),
                    ("highlighted", "noinherit"),
                    ("pointer", "noinherit"),
                ]
            ),
        ).ask()
        if not choices:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            selected_analysts = choices
            print(f"\nSelected analysts: " f"{', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}")

    # Select LLM model based on whether Ollama is being used
    model_name = ""
    model_provider = None

    if args.ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")

        # Select from Ollama-specific models
        model_name = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_name:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        if model_name == "-":
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

        # Ensure Ollama is installed, running, and the model is available
        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            sys.exit(1)

        model_provider = ModelProvider.OLLAMA.value
        print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    else:
        # Use the standard cloud-based LLM selection
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        
        model_name, model_provider = model_choice

        model_info = get_model_info(model_name, model_provider)
        if model_info:
            if model_info.is_custom():
                model_name = questionary.text("Enter the custom model name:").ask()
                if not model_name:
                    print("\n\nInterrupt received. Exiting...")
                    sys.exit(0)

            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    # Create and run the alpha backtester
    backtester = AlphaBacktester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        daily_capital=args.daily_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
    )

    print(f"\n{Fore.CYAN}Alpha-Focused Backtesting Configuration:{Style.RESET_ALL}")
    print(f"Trading Universe: {', '.join(tickers)} (SPY benchmark excluded)")
    print(f"Daily Capital: ${args.daily_capital:,.2f}")
    print(f"Evaluation Method: 30-day forward alpha vs SPY")

    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()
