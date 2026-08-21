import concurrent.futures
import itertools
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SECTOR_MAP_JA = {
    "Technology": "テクノロジー・半導体",
    "Consumer Cyclical": "一般消費財・自動車・小売り",
    "Consumer Defensive": "食品・生活必需品",
    "Industrials": "機械・重工・インフラ",
    "Basic Materials": "素材・化学・鉄鋼",
    "Healthcare": "医薬品・ヘルスケア",
    "Communication Services": "IT・通信・ゲーム・メディア",
    "Energy": "エネルギー・資源",
    "Utilities": "電力・ガス",
    "Real Estate": "不動産"
}

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
    mcap_rank: int = 0

@dataclass
class QuizPair:
    round_id: str
    category: str
    tier: int
    is_same_sector: bool
    company_a: CompanyProfile
    company_b: CompanyProfile
    winner: str
    ebitda_diff_pct: float
    multiple_ratio: float

class UnifiedValuationPipeline:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

    def fetch_jp_ranking(self, target_count: int = 1000) -> List[Tuple[str, str]]:
        company_list = []
        seen = set()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        pages = target_count // 50

        logging.info(f"Yahoo!ファイナンスから日本株上位 {target_count} 社を取得中...")
        for page in range(1, pages + 1):
            url = f"https://finance.yahoo.co.jp/stocks/ranking/marketCapitalHigh?market=prime&page={page}"
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code != 200: continue
                soup = BeautifulSoup(res.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "/quote/" in href and ".T" in href:
                        code = href.split("/quote/")[1].split(".T")[0].strip()
                        raw_name = a.get_text().strip()
                        if len(code) == 4 and code.isdigit() and code not in seen and raw_name:
                            name = re.sub(r"^株式会社|株式会社$|\(株\)|（株）", "", raw_name).strip()
                            company_list.append((code, name))
                            seen.add(code)
                time.sleep(0.05)
            except Exception: pass
        return company_list[:target_count]

    def fetch_jp_single(self, code: str, name: str, rank: int) -> Optional[CompanyProfile]:
        try:
            info = yf.Ticker(f"{code}.T").info
            sector_en = info.get("sector", "")
            if "Financial" in sector_en or "Banking" in sector_en: return None

            mcap_raw, debt_raw = info.get("marketCap"), info.get("totalDebt", 0) or 0
            cash_raw, ebitda_raw = info.get("totalCash", 0) or 0, info.get("ebitda")
            if not mcap_raw or not ebitda_raw or ebitda_raw <= 0: return None

            mcap = mcap_raw / 1e8
            net_debt = (debt_raw - cash_raw) / 1e8
            ev = mcap + net_debt
            ebitda = ebitda_raw / 1e8
            if ev <= 0 or ebitda <= 0: return None

            ev_ebitda = ev / ebitda
            if ev_ebitda < 0.5 or ev_ebitda > 80.0: return None
            sector_ja = SECTOR_MAP_JA.get(sector_en, "その他産業")

            return CompanyProfile(
                ticker=code, name=name, sector=sector_ja, industry=info.get("industry", "N/A"),
                market_cap=round(mcap, 1), net_debt=round(net_debt, 1), ev=round(ev, 1),
                ebitda=round(ebitda, 1), ev_ebitda=round(ev_ebitda, 2),
                description=f"東証プライム時価総額 第{rank}位 / 【{sector_ja}】",
                mcap_rank=rank
            )
        except Exception: return None

    # ---------------------------------------------------------
    # 🇺🇸 修正箇所: URLを "master" から "main" に変更
    # ---------------------------------------------------------
    def fetch_sp500_tickers(self) -> List[Tuple[str, str]]:
        logging.info("S&P 500 構成銘柄データを取得中...")
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        try:
            df = pd.read_csv(url)
            tickers = [(str(row["Symbol"]).replace(".", "-").strip(), str(row["Security"]).strip()) for _, row in df.iterrows()]
            logging.info(f"S&P 500 銘柄数: {len(tickers)} 社")
            return tickers
        except Exception as e:
            logging.error(f"S&P 500 CSV取得失敗: {e}")
            return []

    def fetch_us_single(self, ticker: str, name: str) -> Optional[CompanyProfile]:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "")
            if "Financial" in sector or "Banking" in sector: return None

            mcap_raw, debt_raw = info.get("marketCap"), info.get("totalDebt", 0) or 0
            cash_raw, ebitda_raw = info.get("totalCash", 0) or 0, info.get("ebitda")
            if not mcap_raw or not ebitda_raw or ebitda_raw <= 0: return None

            mcap = mcap_raw / 1e9
            net_debt = (debt_raw - cash_raw) / 1e9
            ev = mcap + net_debt
            ebitda = ebitda_raw / 1e9
            if ev <= 0 or ebitda <= 0: return None

            ev_ebitda = ev / ebitda
            if ev_ebitda < 0.5 or ev_ebitda > 90.0: return None
            industry = info.get("industry", "S&P 500")

            return CompanyProfile(
                ticker=ticker, name=name, sector=sector, industry=industry,
                market_cap=round(mcap, 2), net_debt=round(net_debt, 2), ev=round(ev, 2),
                ebitda=round(ebitda, 2), ev_ebitda=round(ev_ebitda, 2),
                description=f"S&P 500 / {sector} ({industry})"
            )
        except Exception: return None

    def generate_pairs(self, companies: List[CompanyProfile], is_us: bool = False) -> List[QuizPair]:
        pairs = []
        pair_cnt = 1
        sorted_comps = sorted(companies, key=lambda x: x.ebitda)
        n = len(sorted_comps)

        diff_limit = 0.20 if is_us else 0.15
        mult_limit = 1.35 if is_us else 1.4

        for i in range(n):
            a = sorted_comps[i]
            for j in range(i + 1, n):
                b = sorted_comps[j]
                diff_pct = (b.ebitda - a.ebitda) / b.ebitda
                if diff_pct > diff_limit: break
                
                ratio = max(a.ev_ebitda, b.ev_ebitda) / min(a.ev_ebitda, b.ev_ebitda)
                if ratio >= mult_limit:
                    top_rank = min(a.mcap_rank, b.mcap_rank)
                    
                    if is_us:
                        if top_rank <= 50: tier = 1
                        elif top_rank <= 150: tier = 2
                        elif top_rank <= 300: tier = 3
                        else: tier = 4
                        cat = "Same Sector" if a.sector == b.sector else "Cross Sector"
                    else:
                        if top_rank <= 100: tier = 1
                        elif top_rank <= 300: tier = 2
                        elif top_rank <= 500: tier = 3
                        else: tier = 4
                        cat = "同業対決" if a.sector == b.sector else "異業種対決"

                    pairs.append(QuizPair(
                        round_id=f"ROUND_{pair_cnt:04d}", category=cat, tier=tier,
                        is_same_sector=(a.sector == b.sector),
                        company_a=a, company_b=b, winner="A" if a.ev > b.ev else "B",
                        ebitda_diff_pct=round(diff_pct * 100, 1), multiple_ratio=round(ratio, 2)
                    ))
                    pair_cnt += 1
        return pairs

    def run(self):
        logging.info("=== 日本市場パイプライン実行 ===")
        jp_ranking = self.fetch_jp_ranking(1000)
        jp_valid = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fmap = {ex.submit(self.fetch_jp_single, c, n, idx + 1): (c, n) for idx, (c, n) in enumerate(jp_ranking)}
            for f in concurrent.futures.as_completed(fmap):
                r = f.result()
                if r: jp_valid.append(r)
        
        jp_pairs = self.generate_pairs(jp_valid, is_us=False)
        with open("japanese_valuation_quiz.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in jp_pairs], f, ensure_ascii=False, indent=2)
        logging.info(f"日本版データセット生成完了: {len(jp_pairs)} 問")

        logging.info("=== 米国S&P 500パイプライン実行 ===")
        us_tickers = self.fetch_sp500_tickers()
        us_valid = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fmap = {ex.submit(self.fetch_us_single, t, n): (t, n) for t, n in us_tickers}
            for f in concurrent.futures.as_completed(fmap):
                r = f.result()
                if r: us_valid.append(r)

        us_valid.sort(key=lambda x: x.market_cap, reverse=True)
        for idx, comp in enumerate(us_valid):
            comp.mcap_rank = idx + 1
            comp.description = f"S&P 500 Rank #{comp.mcap_rank} / {comp.sector}"

        us_pairs = self.generate_pairs(us_valid, is_us=True)
        with open("us_valuation_quiz.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in us_pairs], f, ensure_ascii=False, indent=2)
        logging.info(f"米国版データセット生成完了: {len(us_pairs)} 問")

if __name__ == "__main__":
    UnifiedValuationPipeline(max_workers=10).run()
