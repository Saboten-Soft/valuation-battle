import concurrent.futures
import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import List, Optional
from bs4 import BeautifulSoup
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class CompanyProfile:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float      # 億円
    net_debt: float        # 億円
    ev: float              # 億円
    ebitda: float          # 億円
    ev_ebitda: float       # 倍
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

class YahooRankingValuationPipeline:
    def __init__(self, target_count: int = 1000, max_workers: int = 12):
        self.target_count = target_count
        self.max_workers = max_workers
        self.companies: List[CompanyProfile] = []

    def fetch_top_tickers_from_yahoo(self) -> List[str]:
        """Yahoo!ファイナンスの時価総額ランキング（プライム）から上位1,000社を取得"""
        tickers = []
        pages_to_fetch = (self.target_count // 50)  # 1ページ50銘柄 × 20ページ = 1,000銘柄
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        logging.info(f"Yahoo!ファイナンスからプライム時価総額上位 {self.target_count} 社を動的取得中...")
        for page in range(1, pages_to_fetch + 1):
            url = f"https://finance.yahoo.co.jp/stocks/ranking/marketCapitalHigh?market=prime&page={page}"
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code != 200:
                    logging.warning(f"Page {page} status code: {res.status_code}")
                    continue

                soup = BeautifulSoup(res.text, "html.parser")
                # 銘柄コードリンクの抽出 (例: /quote/7203.T)
                links = soup.find_all("a", href=True)
                for a in links:
                    href = a["href"]
                    if "/quote/" in href and ".T" in href:
                        code = href.split("/quote/")[1].split(".T")[0].strip()
                        if len(code) == 4 and code.isdigit() and code not in tickers:
                            tickers.append(code)

                time.sleep(0.2)  # サーバー負荷軽減
            except Exception as e:
                logging.warning(f"Failed to scrape page {page}: {e}")

        logging.info(f"ランキングから抽出された銘柄数: {len(tickers)} 社")
        return tickers[:self.target_count]

    def fetch_single_company(self, code: str) -> Optional[CompanyProfile]:
        """個別銘柄の財務データを yfinance から取得"""
        try:
            stock = yf.Ticker(f"{code}.T")
            info = stock.info

            # 金融業（銀行・保険・証券・その他金融）を除外
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
        tickers = self.fetch_top_tickers_from_yahoo()
        logging.info(f"{len(tickers)} 社の財務データを並行取得します...")
        
        valid = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {executor.submit(self.fetch_single_company, code): code for code in tickers}
            for future in concurrent.futures.as_completed(future_to_code):
                res = future.result()
                if res:
                    valid.append(res)

        logging.info(f"有効な非金融企業: {len(valid)} 社を取得")

        # 高速スライディングウィンドウ法によるペア生成
        pairs = []
        pair_cnt = 1
        sorted_comps = sorted(valid, key=lambda x: x.ebitda)
        n = len(sorted_comps)

        for i in range(n):
            a = sorted_comps[i]
            for j in range(i + 1, n):
                b = sorted_comps[j]
                diff_pct = (b.ebitda - a.ebitda) / b.ebitda
                if diff_pct > 0.15:  # EBITDA差 ±15%以内
                    break
                ratio = max(a.ev_ebitda, b.ev_ebitda) / min(a.ev_ebitda, b.ev_ebitda)
                if ratio >= 1.5:   # マルチプル格差 1.5倍以上
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

        logging.info(f"生成されたクイズペア数: {len(pairs)} 問")
        with open("japanese_valuation_quiz.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in pairs], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    YahooRankingValuationPipeline(target_count=1000, max_workers=12).run_pipeline()
