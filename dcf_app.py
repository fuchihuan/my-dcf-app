import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st  # 匯入 Streamlit

# --- 核心假設 (現在變成網頁上的選項了！) ---
ASSUMED_TAX_RATE_FALLBACK = 0.20
ASSUMED_EBIT_MARGIN_FALLBACK = 0.05
# ---

# 讓 pandas 數字格式化
pd.options.display.float_format = '{:,.0f}'.format

# =============================================================================
# --- 核心 DCF 估值函數 ---
# (這個函數跟之前「完全一樣」，我們沒有動它)
# =============================================================================
def run_dcf_model(ticker_symbol, forecast_years, revenue_growth, perpetual_growth, risk_free, market_return):
    """
    執行完整的 DCF 估值模型並返回結果
    """
    try:
        # --- 階段 1: 抓取數據 ---
        st.write(f"--- 階段 1：抓取 {ticker_symbol} 財務數據 ---")
        ticker = yf.Ticker(ticker_symbol)
        
        income_stmt = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        info = ticker.info
        
        if income_stmt.empty or balance_sheet.empty or cash_flow.empty:
            raise ValueError("財務報表數據為空。")
        st.write("✅ 數據抓取成功")

        # --- 階段 2: 計算歷史平均比率 ---
        st.write("--- 階段 2：計算歷史平均比率 ---")
        
        if 'Total Revenue' not in income_stmt.index:
            raise ValueError("找不到 'Total Revenue' (總營收)，模型無法繼續。")
        hist_revenue = income_stmt.loc['Total Revenue'].iloc[:3]

        # 1. EBIT Margin
        if 'Operating Income' in income_stmt.index:
            hist_ebit = income_stmt.loc['Operating Income'].iloc[:3]
            hist_ebit_margin = (hist_ebit / hist_revenue).mean()
        else:
            hist_ebit_margin = ASSUMED_EBIT_MARGIN_FALLBACK
            st.warning(f"找不到 'Operating Income'，使用 {hist_ebit_margin:.1%} 作為假設。")

        # 2. Tax Rate
        hist_effective_tax_rate = None
        tax_key_found = None
        if 'Income Before Tax' in income_stmt.index: tax_key_found = 'Income Before Tax'
        elif 'Pretax Income' in income_stmt.index: tax_key_found = 'Pretax Income'
            
        if tax_key_found and 'Income Tax Expense' in income_stmt.index:
            hist_income_before_tax = income_stmt.loc[tax_key_found].iloc[:3]
            hist_tax_expense = income_stmt.loc['Income Tax Expense'].iloc[:3]
            taxable_years = hist_income_before_tax > 0
            if taxable_years.any():
                hist_effective_tax_rate = (hist_tax_expense[taxable_years] / hist_income_before_tax[taxable_years]).mean()
                hist_effective_tax_rate = max(0, min(hist_effective_tax_rate, 1))
                
        if hist_effective_tax_rate is None:
            hist_effective_tax_rate = ASSUMED_TAX_RATE_FALLBACK
            st.warning(f"找不到 'Income Before Tax'，使用 {hist_effective_tax_rate:.1%} 作為假設稅率。")

        # 3. D&A
        hist_d_and_a = None
        if 'Depreciation And Amortization' in cash_flow.index: hist_d_and_a = cash_flow.loc['Depreciation And Amortization'].iloc[:3]
        elif 'Depreciation' in cash_flow.index: hist_d_and_a = cash_flow.loc['Depreciation'].iloc[:3]
            
        if hist_d_and_a is not None:
            hist_d_and_a_as_pct_rev = (hist_d_and_a / hist_revenue).mean()
        else:
            hist_d_and_a_as_pct_rev = 0; st.warning("找不到 D&A 數據，假設為 0。")

        # 4. CapEx
        if 'Capital Expenditures' in cash_flow.index:
            hist_capex = cash_flow.loc['Capital Expenditures'].abs().iloc[:3]
            hist_capex_as_pct_rev = (hist_capex / hist_revenue).mean()
        else:
            hist_capex_as_pct_rev = 0; st.warning("找不到 CapEx 數據，假設為 0。")

        # 5. NWC
        if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
            hist_nwc = (balance_sheet.loc['Total Current Assets'] - balance_sheet.loc['Total Current Liabilities']).iloc[:3]
            hist_change_in_nwc = hist_nwc.diff(-1).iloc[:2]
            hist_change_in_revenue = hist_revenue.diff(-1).iloc[:2]
            hist_change_in_revenue = hist_change_in_revenue.replace(0, np.nan)
            hist_nwc_change_as_pct_rev_change = (hist_change_in_nwc / hist_change_in_revenue).mean()
            if not np.isfinite(hist_nwc_change_as_pct_rev_change): hist_nwc_change_as_pct_rev_change = 0
        else:
            hist_nwc_change_as_pct_rev_change = 0; st.warning("找不到 NWC 數據，假設 NWC 變動為 0。")
        
        last_revenue = hist_revenue.iloc[0]

        # --- 階段 3: 預測未來 FCFF ---
        st.write("--- 階段 3：預測未來 FCFF ---")
        
        forecast_data = [] 
        projected_fcff = [] 

        for i in range(1, forecast_years + 1):
            projected_revenue = last_revenue * (1 + revenue_growth)
            projected_ebit = projected_revenue * hist_ebit_margin
            projected_nopat = projected_ebit * (1 - hist_effective_tax_rate)
            projected_d_and_a = projected_revenue * hist_d_and_a_as_pct_rev
            projected_capex = projected_revenue * hist_capex_as_pct_rev
            projected_change_in_revenue = projected_revenue - last_revenue
            projected_change_in_nwc = projected_change_in_revenue * hist_nwc_change_as_pct_rev_change
            projected_fcff_value = projected_nopat + projected_d_and_a - projected_capex - projected_change_in_nwc
            
            forecast_data.append({
                'Year': f"Year {i}", 'Revenue': projected_revenue, 'NOPAT': projected_nopat,
                'D&A': projected_d_and_a, 'CapEx': -projected_capex,
                'Change NWC': -projected_change_in_nwc, 'FCFF': projected_fcff_value
            })
            projected_fcff.append(projected_fcff_value)
            last_revenue = projected_revenue
            
        forecast_df = pd.DataFrame(forecast_data).set_index('Year')
        st.write(f"📈 未來 {forecast_years} 年 FCFF 預測：")
        st.dataframe(forecast_df.transpose().style.format("{:,.0f}"))

        # --- 階段 4: 計算 WACC ---
        st.write("--- 階段 4：計算 WACC ---")
        
        beta = info.get('beta', 1.0)
        Re = risk_free + beta * (market_return - risk_free)

        interest_expense = 0
        if 'Interest Expense' in income_stmt.index: interest_expense = abs(income_stmt.loc['Interest Expense'].iloc[0])
        long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else 0
        short_term_debt = balance_sheet.loc['Short Term Debt'].iloc[0] if 'Short Term Debt' in balance_sheet.index else 0
        total_debt = long_term_debt + short_term_debt

        Rd = 0.04 
        if total_debt > 0 and interest_expense > 0: Rd = interest_expense / total_debt
        elif total_debt > 0: st.warning("找不到利息支出，使用預設債務成本 4%")
        else: Rd = 0

        E = info.get('marketCap')
        if E is None: raise ValueError("找不到 'marketCap' (市值)，無法計算 WACC。")
        D = total_debt
        V = E + D
        
        wacc = (E/V) * Re + (D/V) * Rd * (1 - hist_effective_tax_rate)
        st.write(f"✅ WACC (折現率) 計算完成: {wacc:.4%}")
        
        # --- 階段 5 & 6 & 7: 計算最終價值 ---
        st.write("--- 階段 5-7：計算最終股價 ---")
        
        if wacc <= perpetual_growth:
            raise ValueError(f"WACC ({wacc:.2%}) 必須大於永續增長率 ({perpetual_growth:.2%})")

        last_projected_fcff = projected_fcff[-1]
        terminal_value = last_projected_fcff * (1 + perpetual_growth) / (wacc - perpetual_growth)
        
        pv_fcff_list = [projected_fcff[i] / ((1 + wacc) ** (i + 1)) for i in range(forecast_years)]
        total_pv_fcff = sum(pv_fcff_list)
        pv_terminal_value = terminal_value / ((1 + wacc) ** forecast_years)
        enterprise_value = total_pv_fcff + pv_terminal_value
        
        cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
        net_debt = total_debt - cash_and_equivalents
        equity_value = enterprise_value - net_debt
        shares_outstanding = info.get('sharesOutstanding')
        
        if shares_outstanding is None or shares_outstanding == 0:
            raise ValueError("找不到總流通股數 (sharesOutstanding)。")

        implied_price_per_share = equity_value / shares_outstanding
        current_price = info.get('currentPrice', info.get('previousClose'))
        if current_price is None: raise ValueError("找不到 'currentPrice' (目前股價)。")
        
        # --- 顯示最終結果 ---
        st.success("🎉 估值計算完成！")
        
        col1, col2 = st.columns(2) 
        col1.metric("模型預估股價 (Implied Price)", f"{implied_price_per_share:,.2f}")
        col2.metric("目前市場股價 (Current Price)", f"{current_price:,.2f}")
        
        diff_percent = (implied_price_per_share - current_price) / current_price
        if diff_percent > 0.01:
            st.success(f"➡️ 模型結果：目前股價可能低估了 {diff_percent:.2%}")
        elif diff_percent < -0.01:
            st.error(f"➡️ 模型結果：目前股價可能高估了 {abs(diff_percent):.2%}")
        else:
            st.info(f"➡️ 模型結果：目前股價估值合理")
            
        with st.expander("點此查看估值計算細節"):
            st.write(f"企業價值 (EV): {enterprise_value:,.0f}")
            st.write(f"減：淨負債 (Net Debt): {net_debt:,.0f}")
            st.write(f"股權價值 (Equity Value): {equity_value:,.0f}")
            st.write(f"總流通股數: {shares_outstanding:,.0f}")
            
    except Exception as e:
        st.error(f"❌ 執行失敗！發生了錯誤：{e}")


# =============================================================================
# --- (v2) Streamlit 網頁介面 (使用數字輸入框) ---
# =============================================================================

st.title('📈 自動 DCF 估值模型')
st.write('這是一個使用 Python 和 Streamlit 打造的專業版 DCF 估值工具。')

# --- 1. 股票代碼輸入 ---
st.header('1. 輸入股票代碼')
ticker_input = st.text_input('請輸入 Yahoo Finance 的股票代碼 (例如: 2344.TW, AAPL)', '2344.TW')

# --- 2. 核心假設 (v2 - 改用 st.number_input) ---
st.header('2. 調整核心假設')
st.write("請直接在下方欄位輸入你的假設數字 (例如 3% 請輸入 3)。")

col1, col2 = st.columns(2) 

with col1:
    st.subheader("增長假設")
    # 我們要求使用者輸入 3 (代表 3%)，而不是 0.03
    p_revenue_growth_pct = st.number_input('未來營收年增率 (%)', min_value=0.0, max_value=50.0, value=3.0, step=0.5, format="%.1f")
    p_perpetual_growth_pct = st.number_input('永續增長率 (%)', min_value=0.0, max_value=10.0, value=2.5, step=0.1, format="%.1f")
    p_forecast_years = st.number_input('預測年數 (年)', min_value=1, max_value=20, value=5, step=1)

with col2:
    st.subheader("折現率假設")
    p_risk_free_pct = st.number_input('無風險利率 (%)', min_value=0.0, max_value=10.0, value=3.0, step=0.1, format="%.1f")
    p_market_return_pct = st.number_input('市場年化報酬率 (%)', min_value=0.0, max_value=20.0, value=8.0, step=0.5, format="%.1f")


# --- 3. 執行按鈕 ---
st.header('3. 執行估值')

if st.button('🚀 開始估值！', type="primary"):
    if ticker_input:
        with st.spinner('正在抓取財報並執行複雜的 DCF 計算中... 請稍候...'):
            
            # (重要！) 把使用者輸入的 3 (%) 轉換回 0.03 
            p_revenue_growth = p_revenue_growth_pct / 100.0
            p_perpetual_growth = p_perpetual_growth_pct / 100.0
            p_risk_free = p_risk_free_pct / 100.0
            p_market_return = p_market_return_pct / 100.0
            
            # 把轉換後的值，傳入 DCF 函數
            run_dcf_model(
                ticker_symbol=ticker_input,
                forecast_years=p_forecast_years, # 年份不需要轉換
                revenue_growth=p_revenue_growth,
                perpetual_growth=p_perpetual_growth,
                risk_free=p_risk_free,
                market_return=p_market_return
            )
    else:
        st.error('請先輸入股票代碼！')
