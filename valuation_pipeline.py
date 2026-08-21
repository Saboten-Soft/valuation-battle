import concurrent.futures
import io
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

# 外部通信が全て落ちた場合でも絶対に動くS&P 500セーフティネット銘柄群（全業種・全階層）
BACKUP_SP500_TICKERS = [
    ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation"), ("NVDA", "NVIDIA Corporation"),
    ("AMZN", "Amazon.com, Inc."), ("GOOGL", "Alphabet Inc."), ("META", "Meta Platforms, Inc."),
    ("TSLA", "Tesla, Inc."), ("AVGO", "Broadcom Inc."), ("WMT", "Walmart Inc."),
    ("LLY", "Eli Lilly and Company"), ("JNJ", "Johnson & Johnson"), ("PG", "Procter & Gamble"),
    ("ORCL", "Oracle Corporation"), ("COST", "Costco Wholesale"), ("HD", "The Home Depot"),
    ("ABBV", "AbbVie Inc."), ("NFLX", "Netflix, Inc."), ("AMD", "Advanced Micro Devices"),
    ("CRM", "Salesforce, Inc."), ("KO", "The Coca-Cola Company"), ("PEP", "PepsiCo, Inc."),
    ("MRK", "Merck & Co., Inc."), ("LIN", "Linde plc"), ("TMO", "Thermo Fisher Scientific"),
    ("ADBE", "Adobe Inc."), ("CSCO", "Cisco Systems"), ("QCOM", "Qualcomm Incorporated"),
    ("TXN", "Texas Instruments"), ("CAT", "Caterpillar Inc."), ("GE", "GE Aerospace"),
    ("AMAT", "Applied Materials"), ("ISRG", "Intuitive Surgical"), ("DIS", "The Walt Disney Company"),
    ("MCD", "McDonald's Corporation"), ("INTU", "Intuit Inc."), ("AMGN", "Amgen Inc."),
    ("UNP", "Union Pacific"), ("HON", "Honeywell"), ("IBM", "IBM Corporation"),
    ("SBUX", "Starbucks"), ("NKE", "NIKE, Inc."), ("UBER", "Uber Technologies"),
    ("BKNG", "Booking Holdings"), ("LRCX", "Lam Research"), ("PANW", "Palo Alto Networks"),
    ("ADI", "Analog Devices"), ("DE", "Deere & Company"), ("RTX", "RTX Corporation"),
    ("SYK", "Stryker Corporation"), ("BA", "The Boeing Company"), ("VRTX", "Vertex Pharmaceuticals"),
    ("MDLZ", "Mondelez International"), ("LMT", "Lockheed Martin"), ("GILD", "Gilead Sciences"),
    ("ABT", "Abbott Laboratories"), ("DHR", "Danaher Corporation"), ("SNPS", "Synopsys"),
    ("CDNS", "Cadence Design Systems"), ("REGN", "Regeneron Pharmaceuticals"), ("PFE", "Pfizer Inc."),
    ("KLAC", "KLA Corporation"), ("BSX", "Boston Scientific"), ("BMY", "Bristol-Myers Squibb"),
    ("EOG", "EOG Resources"), ("SLB", "Schlumberger"), ("MPC", "Marathon Petroleum"),
    ("PSX", "Phillips 66"), ("VLO", "Valero Energy"), ("COP", "ConocoPhillips"),
    ("XOM", "Exxon Mobil"), ("CVX", "Chevron Corporation"), ("OXY", "Occidental Petroleum"),
    ("MAR", "Marriott International"), ("ABNB", "Airbnb, Inc."), ("LULU", "Lululemon Athletica"),
    ("MNST", "Monster Beverage"), ("DECK", "Deckers Outdoor"), ("ORLY", "O'Reilly Automotive"),
    ("AZO", "AutoZone, Inc."), ("ROST", "Ross Stores"), ("TJX", "The TJX Companies"),
    ("TGT", "Target Corporation"), ("CMG", "Chipotle Mexican Grill"), ("YUM", "Yum! Brands"),
    ("FDX", "FedEx Corporation"), ("UPS", "United Parcel Service"), ("NSC", "Norfolk Southern"),
    ("CSX", "CSX Corporation"), ("WM", "Waste Management"), ("EMR", "Emerson Electric"),
    ("ETN", "Eaton Corporation"), ("PH", "Parker-Hannifin"), ("PCAR", "PACCAR Inc"),
    ("GD", "General Dynamics"), ("TDG", "TransDigm Group"), ("NOC", "Northrop Grumman"),
    ("ENPH", "Enphase Energy"), ("FSLR", "First Solar"), ("ALB", "Albemarle Corporation"),
    ("STLD", "Steel Dynamics"), ("NUE", "Nucor Corporation"), ("FCX", "Freeport-McMoRan"),
    ("DOW", "Dow Inc."), ("ECL", "Ecolab Inc."), ("SHW", "The Sherwin-Williams Company"),
    ("APD", "Air Products and Chemicals"), ("CTAS", "Cintas Corporation"), ("FAST", "Fastenal"),
    ("PAYX", "Paychex, Inc."), ("VRSK", "Verisk Analytics"), ("ODFL", "Old Dominion Freight Line")
]

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
    def __init__(self, max_workers: int = 12):
        self.max_workers = max_workers

    # ---------------------------------------------------------
    # 🇯🇵 日本市場 パイプライン (東証プライム 1,000社)
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
    # 🇺🇸 米国市場 パイプライン (S&P 500 500社)
    # ---------------------------------------------------------
    def fetch_sp500_tickers(self) -> List[Tuple[str, str]]:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        # 1. GitHub Datasets (main ブランチ)
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                tickers = [(str(row["Symbol"]).replace(".", "-").strip(), str(row["Security"]).strip()) for _, row in df.iterrows()]
                logging.info(f"S&P 500 (GitHub CSV) 取得成功: {len(tickers)} 社")
                return tickers
        except Exception as e:
            logging.warning(f"GitHub CSV fetch error: {e}")

        # 2. Wikipedia スクレイピング
        try:
            wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            res = requests.get(wiki_url, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                table = soup.find("table", {"id": "constituents"})
                if table:
                    tickers = []
                    for row in table.find_all("tr")[1:]:
                        cols = row.find_all("td")
                        if len(cols) > 1:
                            t = cols[0].text.strip().replace(".", "-")
                            n = cols[1].text.strip()
                            tickers.append((t, n))
                    if len(tickers) >= 400:
                        logging.info(f"S&P 500 (Wikipedia) 取得成功: {len(tickers)} 社")
                        return tickers
        except Exception as e:
            logging.warning(f"Wikipedia fetch error: {e}")

        # 3. 内蔵セーフティネット
        logging.info(f"S&P 500 (内蔵リスト) を使用: {len(BACKUP_SP500_TICKERS)} 社")
        return BACKUP_SP500_TICKERS

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
            
            clean_name = name if (name and len(name) > 1) else (info.get("shortName") or ticker)
            industry = info.get("industry", "S&P 500")

            return CompanyProfile(
                ticker=ticker, name=clean_name, sector=sector, industry=industry,
                market_cap=round(mcap, 2), net_debt=round(net_debt, 2), ev=round(ev, 2),
                ebitda=round(ebitda, 2), ev_ebitda=round(ev_ebitda, 2),
                description=f"S&P 500 / {sector} ({industry})"
            )
        except Exception: return None

    # ---------------------------------------------------------
    # 対戦ペア生成
    # ---------------------------------------------------------
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
                        if top_rank <= 30: tier = 1
                        elif top_rank <= 80: tier = 2
                        elif top_rank <= 160: tier = 3
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
        # 1. 日本株
        logging.info("=== 日本市場 パイプライン開始 ===")
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
        logging.info(f"日本版クイズ生成完了: {len(jp_pairs)} 問")

        # 2. 米国株
        logging.info("=== 米国市場 パイプライン開始 ===")
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
        logging.info(f"米国版クイズ生成完了: {len(us_pairs)} 問")

if __name__ == "__main__":
    UnifiedValuationPipeline(max_workers=12).run()
