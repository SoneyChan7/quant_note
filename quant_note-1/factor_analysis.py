import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from scipy import stats

class FactorAnalyzer:
    """因子分析器，用于评估因子的有效性"""
    
    def __init__(self):
        self.ic_data = {}
        self.returns_data = {}
    
    def calculate_ic(self, factor_values: pd.Series, forward_returns: pd.Series,
                     method: str = 'spearman') -> float:
        """计算因子值与未来收益的信息系数(IC)
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            method: 相关系数计算方法，可选'pearson'或'spearman'
        """
        if method == 'spearman':
            corr_func = stats.spearmanr
        else:
            corr_func = stats.pearsonr
            
        # 去除空值
        mask = ~(factor_values.isna() | forward_returns.isna())
        if mask.sum() < 2:
            return np.nan
            
        ic, _ = corr_func(factor_values[mask], forward_returns[mask])
        return ic
    
    def calculate_factor_returns(self, factor_values: pd.Series,
                               forward_returns: pd.Series,
                               groups: int = 5) -> pd.Series:
        """计算因子分组收益
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            groups: 分组数量
        """
        # 对因子值进行分组
        labels = pd.qcut(factor_values, groups, labels=False)
        return forward_returns.groupby(labels).mean()
    
    def analyze_factor(self, factor_values: pd.DataFrame,
                      prices: pd.DataFrame,
                      forward_periods: List[int] = [1, 5, 10, 20]) -> Dict:
        """综合分析因子
        
        Args:
            factor_values: 因子值DataFrame，index为时间，columns为股票代码
            prices: 价格DataFrame，index为时间，columns为股票代码
            forward_periods: 未来收益计算周期列表
        """
        results = {}
        
        # 计算各期收益率
        returns = {}
        for period in forward_periods:
            returns[period] = prices.pct_change(period).shift(-period)
        
        # 计算IC
        ic_stats = {}
        for period in forward_periods:
            period_ic = []
            for dt in factor_values.index:
                if dt in returns[period].index:
                    ic = self.calculate_ic(
                        factor_values.loc[dt],
                        returns[period].loc[dt]
                    )
                    period_ic.append(ic)
            
            ic_series = pd.Series(period_ic, index=factor_values.index[:len(period_ic)])
            ic_stats[period] = {
                'IC均值': ic_series.mean(),
                'IC标准差': ic_series.std(),
                'IC_IR': ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan,
                'IC>0占比': (ic_series > 0).mean()
            }
        
        results['IC分析'] = ic_stats
        
        # 计算分组收益
        group_returns = {}
        for period in forward_periods:
            period_returns = []
            for dt in factor_values.index:
                if dt in returns[period].index:
                    group_return = self.calculate_factor_returns(
                        factor_values.loc[dt],
                        returns[period].loc[dt]
                    )
                    period_returns.append(group_return)
            
            if period_returns:
                group_returns[period] = pd.DataFrame(period_returns).mean()
        
        results['分组收益分析'] = group_returns
        
        return results

class FactorPerformance:
    """因子绩效评估"""
    
    @staticmethod
    def calculate_turnover(factor_values: pd.DataFrame,
                          top_n: int) -> pd.Series:
        """计算因子换手率
        
        Args:
            factor_values: 因子值DataFrame
            top_n: 选股数量
        """
        def get_top_stocks(row):
            return set(row.nlargest(top_n).index)
        
        turnover = []
        prev_stocks = None
        
        for dt in factor_values.index:
            curr_stocks = get_top_stocks(factor_values.loc[dt])
            if prev_stocks is not None:
                # 计算换手率：新增和减少的股票数量除以2
                turnover_rate = len(curr_stocks.symmetric_difference(prev_stocks)) / (2 * top_n)
                turnover.append(turnover_rate)
            prev_stocks = curr_stocks
        
        return pd.Series(turnover, index=factor_values.index[1:])
    
    @staticmethod
    def calculate_factor_exposure(factor_values: pd.DataFrame) -> pd.DataFrame:
        """计算因子暴露度
        
        Args:
            factor_values: 因子值DataFrame
        """
        # 对因子值进行标准化
        return factor_values.apply(lambda x: (x - x.mean()) / x.std())
    
    @staticmethod
    def calculate_factor_concentration(factor_values: pd.DataFrame,
                                     top_n: int) -> pd.Series:
        """计算因子集中度
        
        Args:
            factor_values: 因子值DataFrame
            top_n: 选股数量
        """
        def calc_concentration(row):
            total_value = abs(row).sum()
            if total_value == 0:
                return np.nan
            top_value = abs(row.nlargest(top_n)).sum()
            return top_value / total_value
        
        return factor_values.apply(calc_concentration, axis=1)