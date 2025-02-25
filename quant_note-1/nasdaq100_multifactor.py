import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

# 纳斯达克100成分股列表
NASDAQ100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'AVGO',
    'PEP', 'COST', 'CSCO', 'TMUS', 'ADBE', 'TXN', 'CMCSA', 'NFLX', 'AMD',
    'INTC', 'QCOM', 'HON', 'AMGN', 'INTU', 'AMAT', 'ISRG', 'SBUX', 'GILD',
    'ADI', 'MDLZ', 'REGN', 'VRTX', 'BKNG', 'LRCX', 'ADP', 'PANW', 'KLAC',
    'SNPS', 'CDNS', 'CHTR', 'MAR', 'MNST', 'ORLY', 'FTNT', 'CTAS', 'PAYX',
    'MCHP', 'ADSK', 'ABNB', 'KDP', 'ASML', 'MRVL', 'NXPI', 'KHC', 'CSX',
    'DXCM', 'EA', 'AEP', 'IDXX', 'BIIB', 'PCAR', 'FAST', 'XEL', 'ODFL',
    'CPRT', 'ROST', 'WBD', 'SIRI', 'DLTR', 'EXC', 'VRSK', 'ANSS', 'FANG',
    'BKR', 'CTSH', 'CSGP', 'EBAY', 'ZS', 'ILMN', 'TEAM', 'ALGN', 'ENPH',
    'WBA', 'SWKS', 'SGEN', 'SPLK', 'LCID', 'MTCH', 'DDOG', 'CRWD', 'RIVN'
]

def get_stock_data(ticker, start_date, end_date):
    """获取股票数据"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        return data
    except Exception as e:
        print(f"Error downloading {ticker}: {str(e)}")
        return pd.DataFrame()

def get_nasdaq100_tickers():
    """自动获取纳斯达克100指数成分股"""
    try:
        nasdaq100 = yf.Ticker('^NDX')
        holdings = nasdaq100.info.get('holdings', [])
        if not holdings:
            raise ValueError("无法获取纳斯达克100成分股信息")
        return [holding['symbol'] for holding in holdings]
    except Exception as e:
        print(f"获取纳斯达克100成分股失败: {str(e)}")
        return []

def calculate_factors(data, params):
    """计算多个因子值"""
    factors = pd.DataFrame(index=data.index)
    # 动量因子
    factors['momentum'] = data['Close'].pct_change(params['momentum_window'])
    # 波动率因子
    factors['volatility'] = data['Close'].pct_change().rolling(params['vol_window']).std()
    # 成交量因子
    factors['volume'] = data['Volume'] / data['Volume'].rolling(params['vol_window']).mean()
    # RSI因子
    factors['rsi'] = data.ta.rsi(close='Close', length=params['rsi_window'])
    return factors

def normalize_factors(factor_df):
    """对因子值进行标准化"""
    return factor_df.apply(lambda x: stats.zscore(x, nan_policy='omit'))

def combine_factors(normalized_factors, weights):
    """组合多个因子得分"""
    weighted_factors = normalized_factors.mul(pd.Series(weights))
    return weighted_factors.sum(axis=1)

def run_backtest(start_date, end_date, params):
    """运行回测"""
    # 获取最新的纳斯达克100成分股
    tickers = get_nasdaq100_tickers()
    if not tickers:
        print("使用默认的纳斯达克100成分股列表")
        tickers = NASDAQ100_TICKERS
    
    # 获取所有股票数据和因子得分
    all_data = {}
    all_scores = {}
    
    for ticker in NASDAQ100_TICKERS:
        try:
            data = get_stock_data(ticker, start_date, end_date)
            if not data.empty and len(data) > params['momentum_window']:
                # 计算因子值
                factors = calculate_factors(data, params)
                # 删除包含NaN的行
                factors = factors.dropna()
                if not factors.empty:
                    normalized_factors = normalize_factors(factors)
                    scores = combine_factors(normalized_factors, params['weights'])
                    
                    all_data[ticker] = data.loc[factors.index, 'Close']
                    all_scores[ticker] = scores
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    # 转换为DataFrame
    prices_df = pd.DataFrame(all_data)
    scores_df = pd.DataFrame(all_scores)
    
    if prices_df.empty or scores_df.empty:
        print("没有获取到有效的股票数据")
        return pd.Series(), {}
    
    # 确保价格和得分数据使用相同的索引
    common_index = prices_df.index.intersection(scores_df.index)
    prices_df = prices_df.loc[common_index]
    scores_df = scores_df.loc[common_index]
    
    # 计算持仓权重
    def select_top_stocks(row):
        valid_values = row.dropna()
        if len(valid_values) < params['top_n']:
            return pd.Series(0, index=row.index)
        top_stocks = valid_values.nlargest(params['top_n']).index
        weights = pd.Series(1.0/params['top_n'], index=row.index)
        weights[~weights.index.isin(top_stocks)] = 0
        return weights
    
    weights = scores_df.resample(params['rebalance_freq']).last().apply(select_top_stocks, axis=1)
    weights = weights.reindex(prices_df.index, method='ffill')
    
    # 计算每日持仓市值
    portfolio_value = params['init_cash']
    daily_values = [portfolio_value]
    dates = [prices_df.index[0]]
    
    for i in range(1, len(prices_df.index)):
        current_date = prices_df.index[i]
        prev_date = prices_df.index[i-1]
        
        # 获取当前持仓权重
        current_weights = weights.loc[current_date]
        prev_weights = weights.loc[prev_date]
        
        # 计算收益
        returns = prices_df.loc[current_date] / prices_df.loc[prev_date] - 1
        portfolio_return = (returns * current_weights).sum()
        
        # 考虑交易费用
        if not current_weights.equals(prev_weights):
            turnover = abs(current_weights - prev_weights).sum() / 2
            portfolio_value *= (1 + portfolio_return - turnover * params['fees'])
        else:
            portfolio_value *= (1 + portfolio_return)
        
        daily_values.append(portfolio_value)
        dates.append(current_date)
    
    portfolio_values = pd.Series(daily_values, index=dates)
    returns = portfolio_values.pct_change().dropna()
    
    # 计算回测统计
    stats = {
        '年化收益率': (portfolio_values[-1] / portfolio_values[0]) ** (252/len(returns)) - 1,
        '夏普比率': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan,
        '最大回撤': (portfolio_values / portfolio_values.cummax() - 1).min(),
        '波动率': returns.std() * np.sqrt(252)
    }
    
    return portfolio_values, stats

if __name__ == '__main__':
    # 策略参数
    params = {
        'momentum_window': 20,
        'vol_window': 20,
        'rsi_window': 14,
        'weights': {
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.2,
            'rsi': 0.3
        },
        'top_n': 5,
        'rebalance_freq': 'W',
        'init_cash': 100000,
        'fees': 0.001
    }
    
    # 运行回测
    portfolio_values, stats = run_backtest(
        start_date='2023-01-01',
        end_date='2024-02-26',
        params=params
    )
    
    # 打印回测统计结果
    print("\n回测统计结果:")
    for key, value in stats.items():
        print(f"{key}: {value:.2%}")
    
    # 可视化回测结果
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values.values)
    plt.title('纳斯达克100多因子策略回测结果')
    plt.xlabel('日期')
    plt.ylabel('组合价值')
    plt.grid(True)
    plt.show()
    
    # 计算并绘制回撤
    drawdown = (portfolio_values / portfolio_values.cummax() - 1)
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown.index, drawdown.values)
    plt.title('策略回撤')
    plt.xlabel('日期')
    plt.ylabel('回撤')
    plt.grid(True)
    plt.show()