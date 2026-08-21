import concurrent.futures
import itertools
import json
import logging
import io
import urllib.request
from dataclasses import asdict, dataclass
from typing import List, Optional
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class CompanyProfile:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    net_debt: float
    ev: float
    ebitda: float
    ev_ebitda: float
    description: str

@dataclass
class QuizPair:
    round_id: str
    category: str
    company_a: CompanyProfile
    company_b: CompanyProfile
    winner: str
    ebitda_diff_pct: float
    multiple_ratio: float

class LargeScaleValuationPipeline:
    JPX_URL = "https://www.jpx.co.jp/markets/statistics-quotes/stocks/tvdivq0000003005-att/data_j.xls"

    def __init__(self, max_companies: int = 1000, max_workers: int = 10):
        self.max_companies = max_companies
        self.max_workers = max_workers
        self.companies: List[CompanyProfile] = []

    def fetch_prime_tickers(self) -> List[str]:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(self.JPX_URL, headers=headers)
        with urllib.request.urlopen(req) as response:
            df = pd.read_excel(io.BytesIO(response.read()))
        df.columns = [str(col).strip() for col in df.columns]
        prime_df = df[df["市場・商品区分"].str.contains("プライム", na=False)].copy()
        financial_sectors = ["銀行業", "証券、商品先物取引業", "保険業", "その他金融業"]
        non_financial_df = prime_df[~prime_df["33業種区分"].isin(financial_sectors)]
        return [str(code).strip() for code in non_financial_df["コード"] if len(str(code).strip()) == 4]

    def fetch_single_company(self, code: str) -> Optional[CompanyProfile]:
        try:
            stock = yf.Ticker(f"{code}.T")
            info = stock.info
            sector = info.get("sector", "")
            if "Financial" in sector or "Banking" in sector:
                return None

            name = info.get("shortName") or info.get("longName") or code
            mcap_raw = info.get("marketCap")
            debt_raw = info.get("totalDebt", 0) or 0
            cash_raw = info.get("totalCash", 0) or 0
            ebitda_raw = info.get("ebitda")

            if not mcap_raw or not ebitda_raw or ebitda_raw <= 0:
                return None

            mcap = mcap_raw / 1e8
            net_debt = (debt_raw - cash_raw) / 1e8
            ev = mcap + net_debt
            ebitda = ebitda_raw / 1e8

            if ev <= 0 or ebitda <= 0:
                return None

            ev_ebitda = ev / ebitda
            if ev_ebitda < 0.5 or ev_ebitda > 80.0:
                return None

            summary = info.get("longBusinessSummary", "")
            desc = summary[:80] + "..." if len(summary) > 80 else summary

            return CompanyProfile(
                ticker=code,
                name=name,
                sector=sector,
                industry=info.get("industry", "N/A"),
                market_cap=round(mcap, 1),
                net_debt=round(net_debt, 1),
                ev=round(ev, 1),
                ebitda=round(ebitda, 1),
                ev_ebitda=round(ev_ebitda, 2),
                description=desc
            )
        except Exception:
            return None

    def run_pipeline(self):
        raw_tickers = self.fetch_prime_tickers()
        valid = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {executor.submit(self.fetch_single_company, code): code for code in raw_tickers}
            for future in concurrent.futures.as_completed(future_to_code):
                res = future.result()
                if res:
                    valid.append(res)

        valid.sort(key=lambda x: x.market_cap, reverse=True)
        self.companies = valid[:self.max_companies]

        # 高速スライディングウィンドウ法によるペア抽出
        pairs = []
        pair_cnt = 1
        sorted_comps = sorted(self.companies, key=lambda x: x.ebitda)
        n = len(sorted_comps)

        for i in range(n):
            a = sorted_comps[i]
            for j in range(i + 1, n):
                b = sorted_comps[j]
                diff_pct = (b.ebitda - a.ebitda) / b.ebitda
                if diff_pct > 0.15:
                    break
                ratio = max(a.ev_ebitda, b.ev_ebitda) / min(a.ev_ebitda, b.ev_ebitda)
                if ratio >= 1.5:
                    pairs.append(QuizPair(
                        round_id=f"ROUND_{pair_cnt:04d}",
                        category="同業対決" if a.sector == b.sector else "異業種対決",
                        company_a=a,
                        company_b=b,
                        winner="A" if a.ev > b.ev else "B",
                        ebitda_diff_pct=round(diff_pct * 100, 1),
                        multiple_ratio=round(ratio, 2)
                    ))
                    pair_cnt += 1

        with open("japanese_valuation_quiz.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in pairs], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    LargeScaleValuationPipeline().run_pipeline()
