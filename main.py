import matplotlib.pyplot as plt
import seaborn as sns

from app import dynamic_plots as plot
from lib.utils import simulation as sim, helper as help
from lib.ticker import Ticker
from lib.portfolio import Portfolio


def main():
    years = 5
    asx = True

    holdings_dict = {
        "QUAL": 100,
        "VAS": 100,
        "CBA": 100,
        "QAN": 333,
        "BHP": 150,    
        "WBC": 200,    
        "NAB": 150,    
        "ANZ": 120,    
        "WOW": 80,     
        "WES": 90,    
        "TLS": 250,  
        "CSL": 50,    
        "MQG": 70,   
        "RIO": 60,    
    }

    portfolio = Portfolio(holdings=holdings_dict, years=years, asx=asx)

    for ticker in portfolio.tickers:
        print(f"{ticker.code} annualized returns = {ticker.get_estimated_annualized_returns()}")
        print(f"{ticker.code} annualized volatility = {ticker.get_estimated_annualized_volatility()}")
        print()

    simulation_df = portfolio.simulate_portfolio_optimization(
        portfolios=10000, sims=100
    )

    fig = plot.plot_portfolio_optimization(portfolio, simulation_df)
    fig.show()


if __name__ == "__main__":
    main()
