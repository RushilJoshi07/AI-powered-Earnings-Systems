import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import time

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(BASE_DIR, "data", "raw", "earnings_calls.pkl")

SP500_TICKERS = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AMP", "AME", "AMGN", "APH", "ADI", "ANSS",
    "AON", "APA", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ",
    "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL",
    "BAC", "BK", "BBWI", "BAX", "BDX", "BRK.B", "BBY", "BIO", "TECH", "BIIB",
    "BLK", "BX", "BA", "BCR", "BKNG", "BWA", "BSX", "BMY", "AVGO", "BR",
    "BRO", "BF.B", "BLDR", "BG", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH",
    "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "COR",
    "CNC", "CNX", "CDAY", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB",
    "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS",
    "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG", "COP", "ED", "STZ", "CEG",
    "COO", "CPRT", "GLW", "CTVA", "CSGP", "COST", "CTRA", "CCI", "CSX", "CMI",
    "CVS", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY", "DVN", "DXCM", "FANG",
    "DLR", "DFS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DHI", "DTE",
    "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "ELV",
    "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR",
    "ESS", "EL", "ETSY", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR",
    "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR",
    "FE", "FI", "FLT", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN",
    "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS",
    "GM", "GPC", "GILD", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC",
    "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM",
    "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW", "INCY",
    "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG", "INTU", "ISRG", "IVZ",
    "INVH", "IQV", "IRM", "JBAL", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM",
    "JNPR", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KLAC",
    "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LIN",
    "LYV", "LKQ", "LMT", "L", "LOW", "LULU", "LYB", "MTB", "MRO", "MPC",
    "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK",
    "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA",
    "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS",
    "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM", "NWSA", "NWS", "NEE",
    "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA",
    "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS",
    "PCAR", "PKG", "PANW", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE",
    "PCG", "PM", "PSX", "PNW", "PXD", "PNC", "POOL", "PPG", "PPL", "PFG",
    "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "QRVO", "PWR",
    "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG",
    "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC",
    "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA", "SOLV",
    "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SYF", "SNPS",
    "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX",
    "TER", "TSLA", "TXN", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV",
    "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL",
    "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VRSN", "VRSK", "VZ", "VRTX",
    "VTRS", "VICI", "V", "VMC", "WRB", "GWW", "WAB", "WBA", "WMT", "DIS",
    "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WRK", "WY",
    "WHR", "WMB", "WTW", "GWW", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS"
]


def load_dataset():
    df = pd.read_pickle(PKL_PATH)
    return df


def parse_date(date_str):
    try:
        date_part = date_str.split(",")[0] + "," + date_str.split(",")[1]
        date_part = date_part.strip()
        return datetime.strptime(date_part, "%b %d, %Y")
    except Exception:
        return None


def get_stock_movement(symbol, earnings_date):
    try:
        start = earnings_date - timedelta(days=3)
        end = earnings_date + timedelta(days=6)

        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False
        )

        if df.empty:
            return None

        prices = df["Close"].dropna()

        if len(prices) < 2:
            return None

        price_on_date = float(prices.iloc[0].iloc[0]) if hasattr(prices.iloc[0], 'iloc') else float(prices.iloc[0])
        price_3days_later = float(prices.iloc[-1].iloc[0]) if hasattr(prices.iloc[-1], 'iloc') else float(prices.iloc[-1])

        movement_pct = ((price_3days_later - price_on_date) / price_on_date) * 100
        label = 1 if price_3days_later > price_on_date else 0

        return {
            "price_on_earnings_date": round(price_on_date, 2),
            "price_3days_later": round(price_3days_later, 2),
            "movement_pct": round(movement_pct, 2),
            "label": label
        }

    except Exception:
        return None

def get_excess_return(symbol, earnings_date):
    try:
        start = earnings_date - timedelta(days=3)
        end = earnings_date + timedelta(days=6)

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        stock_df = yf.download(symbol, start=start_str, end=end_str, progress=False)
        spy_df = yf.download("SPY", start=start_str, end=end_str, progress=False)

        if stock_df.empty or spy_df.empty:
            return None, None, None, None

        stock_prices = stock_df["Close"].dropna()
        spy_prices = spy_df["Close"].dropna()

        if len(stock_prices) < 2 or len(spy_prices) < 2:
            return None, None, None, None

        if hasattr(stock_prices.iloc[0], 'iloc'):
            stock_prices = stock_prices.iloc[:, 0]
        if hasattr(spy_prices.iloc[0], 'iloc'):
            spy_prices = spy_prices.iloc[:, 0]

        stock_return = (float(stock_prices.iloc[-1]) - float(stock_prices.iloc[0])) / float(stock_prices.iloc[0]) * 100
        spy_return = (float(spy_prices.iloc[-1]) - float(spy_prices.iloc[0])) / float(spy_prices.iloc[0]) * 100
        excess_return = stock_return - spy_return
        label = 1 if excess_return > 0 else 0

        return round(stock_return, 4), round(spy_return, 4), round(excess_return, 4), label

    except Exception:
        return None, None, None, None

def collect_training_data(sample_size=3600):
    df = load_dataset()
    df = df[df['ticker'].isin(SP500_TICKERS)]
    print(f"total S&P 500 records available: {len(df)}")

    # load existing records so we do not reprocess them
    existing_path = os.path.join(BASE_DIR, "data", "raw", "training_data.csv")
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        existing_keys = set(zip(existing["symbol"], existing["date"]))
        print(f"already have {len(existing)} records, skipping those")
    else:
        existing = pd.DataFrame()
        existing_keys = set()

    # shuffle for diversity
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)

    all_records = []
    failed = 0

    for idx, row in df.iterrows():
        if len(all_records) + len(existing) >= sample_size:
            break

        ticker = row['ticker']
        date_str = row['date']
        transcript = row['transcript']
        quarter = row['q']

        earnings_date = parse_date(date_str)
        if earnings_date is None:
            failed += 1
            continue

        date_clean = earnings_date.strftime("%Y-%m-%d")

        # skip if we already have this record
        if (ticker, date_clean) in existing_keys:
            continue

        # get excess return instead of raw return
        result = get_excess_return(ticker, earnings_date)
        if result[0] is None:
            failed += 1
            continue

        stock_return, spy_return, excess_return, label = result

        record = {
            "symbol": ticker,
            "date": date_clean,
            "quarter": quarter,
            "content": transcript,
            "price_on_earnings_date": None,
            "price_3days_later": None,
            "movement_pct": stock_return,
            "excess_return": excess_return,
            "spy_return": spy_return,
            "label": label
        }

        all_records.append(record)

        if len(all_records) % 50 == 0:
            print(f"collected {len(all_records)} new records")

        time.sleep(0.1)

    new_df = pd.DataFrame(all_records)

    # combine with existing records
    if not existing.empty:
        # make sure columns align
        for col in new_df.columns:
            if col not in existing.columns:
                existing[col] = None
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = None
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="first")
    os.makedirs("data/raw", exist_ok=True)
    combined.to_csv(existing_path, index=False)

    print(f"total records saved: {len(combined)}")
    print(f"new records added: {len(new_df)}")
    print(f"failed or skipped: {failed}")
    print(f"label distribution: {combined['label'].value_counts().to_dict()}")

    return combined

def fetch_live_transcript(symbol):
    try:
        df = load_dataset()
        company_data = df[df['ticker'] == symbol.upper()].sort_values('date', ascending=False)

        if not company_data.empty:
            latest = company_data.iloc[0]
            earnings_date = parse_date(latest['date'])

            return {
                "symbol": symbol.upper(),
                "date": earnings_date.strftime("%Y-%m-%d") if earnings_date else latest['date'],
                "quarter": latest['q'],
                "content": latest['transcript']
            }

        from earningscall import get_company
        company = get_company(symbol.upper())
        events = company.events()

        if not events:
            return None

        latest_event = events[0]
        transcript = company.get_transcript(year=latest_event.year, quarter=latest_event.quarter)

        if transcript is None or not transcript.text:
            return None

        return {
            "symbol": symbol.upper(),
            "date": latest_event.conference_date.strftime("%Y-%m-%d"),
            "quarter": f"{latest_event.year}-Q{latest_event.quarter}",
            "content": transcript.text
        }

    except Exception as e:
        print(f"error fetching transcript for {symbol}: {e}")
        return None


if __name__ == "__main__":
    collect_training_data(sample_size=3600)