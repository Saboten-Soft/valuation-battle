import concurrent.futures
import itertools
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# セクターの日本語翻訳マップ
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
    name: str              # 日本語企業名
    sector: str            # 日本語セクター
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

    def clean_company_name(self, raw_name: str) -> str:
        """(株) や 株式会社 などの表記をすっきりトリミング"""
        name = raw_name.replace("(株)", "").replace("（株）", "")
        name = re.sub(r"^株式会社", "", name)
        name = re.sub(r"株式会社$", "", name)
        return name.strip()

    def fetch_top_companies_from_yahoo(self) -> List[Tuple[str, str]]:
        """Yahoo!ファイナンスから『コード』と『日本語社名』を同時に抽出"""
        company_list = []
        seen_codes = set()
        pages_to_fetch = (self.target_count // 50)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        logging.info(f"Yahoo!ファイナンスから時価総額上位 {self.target_count} 社（日本語名）を取得中...")
        for page in range(1, pages_to_fetch + 1):
            url = f"https://finance.yahoo.co.jp/stocks/ranking/marketCapitalHigh?market=prime&page={page}"
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code != 200:
                    continue

                soup = BeautifulSoup(res.text, "html.parser")
                links = soup.find_all("a", href=True)
                for a in links:
                    href = a["href"]
                    if "/quote/" in href and ".T" in href:
                        code = href.split("/quote/")[1].split(".T")[0].strip()
                        raw_name = a.get_text().strip()

                        if len(code) == 4 and code.isdigit() and code not in seen_codes and raw_name:
                            cleaned_name = self.clean_company_name(raw_name)
                            company_list.append((code, cleaned_name))
                            seen_codes.add(code)

                time.sleep(0.15)
            except Exception as e:
                logging.warning(f"Page {page} error: {e}")

        logging.info(f"ランキングから抽出完了: {len(company_list)} 社")
        return company_list[:self.target_count]

    def fetch_single_company(self, code: str, ja_name: str) -> Optional[CompanyProfile]:
        """個別銘柄の財務データを取得して日本語フォーマットへ成形"""
        try:
            stock = yf.Ticker(f"{code}.T")
            info = stock.info

            # 金融業（銀行・保険・証券）を除外
            sector_en = info.get("sector", "")
            if "Financial" in sector_en or "Banking" in sector_en:
                return None

            sector_ja = SECTOR_MAP_JA.get(sector_en, "その他産業")
            industry_en = info.get("industry", "プライム上場企業")

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

            desc_ja = f"東証プライム上場 / 【{sector_ja}】分野の主力企業。"

            return CompanyProfile(
                ticker=code,
                name=ja_name,
                sector=sector_ja,
                industry=industry_en,
                market_cap=round(mcap, 1),
                net_debt=round(net_debt, 1),
                ev=round(ev, 1),
                ebitda=round(ebitda, 1),
                ev_ebitda=round(ev_ebitda, 2),
                description=desc_ja
            )
        except Exception:
            return None

    def run_pipeline(self):
        raw_companies = self.fetch_top_companies_from_yahoo()
        logging.info(f"{len(raw_companies)} 社の財務データを並行処理します...")

        valid = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self.fetch_single_company, code, name): (code, name)
                for code, name in raw_companies
            }
            for future in concurrent.futures.as_completed(future_map):
                res = future.result()
                if res:
                    valid.append(res)

        logging.info(f"有効企業数: {len(valid)} 社（日本語データ化完了）")

        # 高速スライディングウィンドウ法による対戦ペア生成
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
                if ratio >= 1.5:   # マルチプル差 1.5倍以上
                    category = "同業対決" if a.sector == b.sector else "異業種対決"
                    pairs.append(QuizPair(
                        round_id=f"ROUND_{pair_cnt:04d}",
                        category=category,
                        company_a=a,
                        company_b=b,
                        winner="A" if a.ev > b.ev else "B",
                        ebitda_diff_pct=round(diff_pct * 100, 1),
                        multiple_ratio=round(ratio, 2)
                    ))
                    pair_cnt += 1

        logging.info(f"生成された日本語クイズペア: {len(pairs)} 問")
        with open("japanese_valuation_quiz.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in pairs], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    YahooRankingValuationPipeline(target_count=1000, max_workers=12).run_pipeline()
