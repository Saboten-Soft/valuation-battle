import concurrent.futures
import io
import itertools
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
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
    market_cap: float      # 日本: 億円 / 米国: $B
    net_debt: float        # 日本: 億円 / 米国: $B
    ev: float              # 日本: 億円 / 米国: $B
    ebitda: float          # 日本: 億円 / 米国: $B
    ev_ebitda: float       # 倍
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

class FullScaleValuationPipeline:
    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers

    # ---------------------------------------------------------
    # 🇯🇵 日本市場: 東証プライム時価総額 上位1,000社
    # ---------------------------------------------------------
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
                time.sleep(0.08)
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
    # 🇺🇸 米国市場: S&P 500 全構成銘柄 (約500社)
    # ---------------------------------------------------------
    def fetch_sp500_tickers(self) -> List[Tuple[str, str]]:
        """Wikipedia + GitHub Raw の二重化で確実に全500社を取得"""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        tickers = []

        # 1. Wikipediaから取得を試行
        try:
            wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            res = requests.get(wiki_url, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                table = soup.find("table", {"id": "constituents"})
                if table:
                    for row in table.find_all("tr")[1:]:
                        cols = row.find_all("td")
                        if len(cols) > 1:
                            t = cols[0].text.strip().replace(".", "-")
                            n = cols[1].text.strip()
                            tickers.append((t, n))
        except Exception as e:
            logging.warning(f"Wikipedia fetch failed: {e}")

        # 2. フォールバック: GitHub Datasetsオープンデータ
        if len(tickers) < 400:
            logging.info("フォールバックデータソースからS&P 500を取得中...")
            try:
                fb_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
                df = pd.read_csv(fb_url)
                tickers = [(row["Symbol"].replace(".", "-"), row["Security"]) for _, row in df.iterrows()]
            except Exception as e:
                logging.error(f"Fallback fetch failed: {e}")

        logging.info(f"S&P 500 取得対象数: {len(tickers)} 社")
        return tickers

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
            
            clean_name = name if name else (info.get("shortName") or ticker)
            industry = info.get("industry", "S&P 500")

            return CompanyProfile(
                ticker=ticker, name=clean_name, sector=sector, industry=industry,
                market_cap=round(mcap, 2), net_debt=round(net_debt, 2), ev=round(ev, 2),
                ebitda=round(ebitda, 2), ev_ebitda=round(ev_ebitda, 2),
                description=f"S&P 500 / {sector} ({industry})"
            )
        except Exception: return None

    # ---------------------------------------------------------
    # ペア生成 & ティア判定ロジック
    # ---------------------------------------------------------
    def generate_pairs(self, companies: List[CompanyProfile], is_us: bool = False) -> List[QuizPair]:
        pairs = []
        pair_cnt = 1
        sorted_comps = sorted(companies, key=lambda x: x.ebitda)
        n = len(sorted_comps)

        for i in range(n):
            a = sorted_comps[i]
            for j in range(i + 1, n):
                b = sorted_comps[j]
                diff_pct = (b.ebitda - a.ebitda) / b.ebitda
                if diff_pct > 0.15: break # EBITDA差 ±15%以内
                
                ratio = max(a.ev_ebitda, b.ev_ebitda) / min(a.ev_ebitda, b.ev_ebitda)
                if ratio >= 1.4: # マルチプル差 1.4倍以上
                    top_rank = min(a.mcap_rank, b.mcap_rank)
                    
                    if is_us:
                        if top_rank <= 50: tier = 1       # Top 50 Mega-Cap
                        elif top_rank <= 150: tier = 2    # 51-150 Core Large
                        elif top_rank <= 300: tier = 3    # 151-300 Mid-Cap
                        else: tier = 4                    # 301-500 Niche / Boss
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
        # 1. 日本株 1,000社
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

        # 2. 米国株 S&P 500全社
        logging.info("=== 米国S&P 500パイプライン実行 ===")
        us_tickers = self.fetch_sp500_tickers()
        us_valid = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fmap = {ex.submit(self.fetch_us_single, t, n): (t, n) for t, n in us_tickers}
            for f in concurrent.futures.as_completed(fmap):
                r = f.result()
                if r: us_valid.append(r)

        # 時価総額順にランク付け
        us_valid.sort(key=lambda x: x.market_cap, reverse=True)
        for idx, comp in enumerate(us_valid):
            comp.mcap_rank = idx + 1
            comp.description = f"S&P 500 Rank #{comp.mcap_rank} / {comp.sector}"

        us_pairs = self.generate_pairs(us_valid, is_us=True)
        with open("us_valuation_quiz.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in us_pairs], f, ensure_ascii=False, indent=2)
        logging.info(f"米国版データセット生成完了: {len(us_pairs)} 問")

if __name__ == "__main__":
    FullScaleValuationPipeline(max_workers=16).run()
