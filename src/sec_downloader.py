"""
SEC filings downloader and parser
Downloads and extracts MD&A and Risk sections from 10-K and 10-Q filings
"""

import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


class SECFilingsDownloader:
    """Downloads and parses SEC filings from EDGAR API"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://www.sec.gov/Archives/edgar"
        self.api_url = "https://data.sec.gov/api/xbrl/companyconcept"
        self.search_url = "https://data.sec.gov/submissions"
        self.headers = {
            "User-Agent": "Market Risk System research@example.com",
            "Accept": "application/json",
        }
        self.rate_limit_delay = 0.1  # SEC allows 10 requests per second

    def get_company_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a company symbol

        Args:
            symbol: Stock symbol

        Returns:
            CIK string if found, None otherwise
        """
        try:
            # Use company tickers API
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(tickers_url, headers=self.headers)

            if response.status_code == 200:
                tickers_data = response.json()

                for entry in tickers_data.values():
                    if entry.get("ticker", "").upper() == symbol.upper():
                        cik = str(entry.get("cik_str", "")).zfill(10)
                        self.logger.info(f"Found CIK {cik} for {symbol}")
                        return cik

            self.logger.warning(f"CIK not found for {symbol}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting CIK for {symbol}: {str(e)}")
            return None

    def get_company_filings(
        self, cik: str, form_types: List[str] = ["10-K", "10-Q"]
    ) -> List[Dict]:
        """
        Get list of filings for a company

        Args:
            cik: Company CIK
            form_types: List of form types to retrieve

        Returns:
            List of filing dictionaries
        """
        try:
            url = f"{self.search_url}/CIK{cik}.json"
            response = requests.get(url, headers=self.headers)

            if response.status_code != 200:
                self.logger.error(
                    f"Failed to get filings for CIK {cik}: {response.status_code}"
                )
                return []

            data = response.json()
            filings = []

            # Get recent filings
            recent_filings = data.get("filings", {}).get("recent", {})

            if not recent_filings:
                return []

            accession_numbers = recent_filings.get("accessionNumber", [])
            filing_dates = recent_filings.get("filingDate", [])
            forms = recent_filings.get("form", [])

            for i, form in enumerate(forms):
                if (
                    form in form_types
                    and i < len(accession_numbers)
                    and i < len(filing_dates)
                ):
                    filings.append(
                        {
                            "cik": cik,
                            "form_type": form,
                            "accession_number": accession_numbers[i],
                            "filing_date": filing_dates[i],
                            "url": f"{self.base_url}/data/{cik.replace('-', '')}/{accession_numbers[i].replace('-', '')}/{accession_numbers[i]}-index.html",
                        }
                    )

            # Sort by filing date (most recent first)
            filings.sort(key=lambda x: x["filing_date"], reverse=True)

            self.logger.info(f"Found {len(filings)} filings for CIK {cik}")
            return filings

        except Exception as e:
            self.logger.error(f"Error getting filings for CIK {cik}: {str(e)}")
            return []

    def download_filing(self, accession_number: str, cik: str) -> Optional[str]:
        """
        Download a specific filing

        Args:
            accession_number: Filing accession number
            cik: Company CIK

        Returns:
            Filing HTML content if successful, None otherwise
        """
        try:
            # Clean accession number and CIK for URL
            clean_accession = accession_number.replace("-", "")
            clean_cik = cik.replace("-", "")

            # Try different URL patterns
            possible_urls = [
                f"{self.base_url}/data/{clean_cik}/{clean_accession}/{accession_number}.txt",
                f"{self.base_url}/data/{clean_cik}/{clean_accession}/{clean_cik}-{accession_number}.txt",
                f"{self.base_url}/data/{clean_cik}/{clean_accession}/primary-document.html",
            ]

            for url in possible_urls:
                try:
                    response = requests.get(url, headers=self.headers)

                    if response.status_code == 200:
                        self.logger.info(f"Downloaded filing from {url}")
                        return response.text

                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    continue

            self.logger.warning(f"Could not download filing {accession_number}")
            return None

        except Exception as e:
            self.logger.error(f"Error downloading filing {accession_number}: {str(e)}")
            return None

    def extract_sections(
        self, filing_content: str, sections: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Extract specific sections from SEC filing

        Args:
            filing_content: Raw filing HTML/text content
            sections: List of sections to extract (default: MD&A and Risk Factors)

        Returns:
            Dictionary with section name as key and content as value
        """
        if sections is None:
            sections = [
                "Management's Discussion and Analysis",
                "MD&A",
                "Risk Factors",
                "Business Risks",
                "Risk Management",
            ]

        extracted = {}

        try:
            # Clean HTML if present
            if "<html" in filing_content.lower():
                soup = BeautifulSoup(filing_content, "html.parser")
                text = soup.get_text()
            else:
                text = filing_content

            # Common patterns for section headers
            patterns = {
                "mda": [
                    r"(?i)(item\s*[27]\.?\s*management'?s\s*discussion\s*and\s*analysis.*?)(item\s*[38]|\n\s*item|\Z)",
                    r"(?i)(management'?s\s*discussion\s*and\s*analysis.*?)(risk\s*factors|\n\s*item|\Z)",
                    r"(?i)(md&a.*?)(risk\s*factors|\n\s*item|\Z)",
                ],
                "risk_factors": [
                    r"(?i)(item\s*1a\.?\s*risk\s*factors.*?)(item\s*[12b]|\n\s*item|\Z)",
                    r"(?i)(risk\s*factors.*?)(unresolved\s*staff\s*comments|\n\s*item|\Z)",
                    r"(?i)(business\s*risks.*?)(\n\s*item|\Z)",
                ],
            }

            # Extract MD&A section
            for pattern in patterns["mda"]:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    extracted["mda"] = self._clean_text(match.group(1))
                    break

            # Extract Risk Factors section
            for pattern in patterns["risk_factors"]:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    extracted["risk_factors"] = self._clean_text(match.group(1))
                    break

            # If specific patterns fail, try more general extraction
            if not extracted:
                lines = text.split("\n")
                current_section = None
                section_content: List[str] = []

                for line in lines:
                    line = line.strip()

                    # Check if line is a section header
                    for section in sections:
                        if section.lower() in line.lower() and len(line) < 200:
                            if section_content and current_section:
                                extracted[current_section] = "\n".join(section_content)

                            current_section = (
                                section.lower().replace("'", "").replace(" ", "_")
                            )
                            section_content = []
                            break

                    if current_section and line:
                        section_content.append(line)

                        # Stop if we hit the next major section
                        if any(
                            stop_word in line.lower()
                            for stop_word in [
                                "item 2",
                                "item 3",
                                "item 4",
                                "controls and procedures",
                            ]
                        ):
                            extracted[current_section] = "\n".join(section_content)
                            current_section = None
                            section_content = []

                # Add final section
                if current_section and section_content:
                    extracted[current_section] = "\n".join(section_content)

            self.logger.info(f"Extracted {len(extracted)} sections from filing")
            return extracted

        except Exception as e:
            self.logger.error(f"Error extracting sections: {str(e)}")
            return {}

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove page numbers and headers
        text = re.sub(r"page\s+\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"table\s+of\s+contents", "", text, flags=re.IGNORECASE)

        # Remove excessive punctuation
        text = re.sub(r"\.{3,}", "...", text)
        text = re.sub(r"-{3,}", "---", text)

        return text.strip()

    def fetch_filings_for_symbols(
        self, symbols: List[str], max_filings_per_symbol: int = 5
    ) -> pd.DataFrame:
        """
        Fetch SEC filings for multiple symbols

        Args:
            symbols: List of stock symbols
            max_filings_per_symbol: Maximum number of filings to fetch per symbol

        Returns:
            DataFrame with filing data and extracted text
        """
        all_filings = []

        for symbol in symbols:
            try:
                self.logger.info(f"Processing filings for {symbol}")

                # Get company CIK
                cik = self.get_company_cik(symbol)
                if not cik:
                    continue

                # Get list of filings
                filings = self.get_company_filings(cik)

                # Process recent filings
                for filing in filings[:max_filings_per_symbol]:
                    self.logger.info(
                        f"Processing {filing['form_type']} for {symbol} dated {filing['filing_date']}"
                    )

                    # Download filing
                    content = self.download_filing(filing["accession_number"], cik)

                    if content:
                        # Extract sections
                        sections = self.extract_sections(content)

                        # Create record
                        record = {
                            "symbol": symbol,
                            "cik": cik,
                            "form_type": filing["form_type"],
                            "filing_date": filing["filing_date"],
                            "accession_number": filing["accession_number"],
                            "mda_text": sections.get("mda", ""),
                            "risk_factors_text": sections.get("risk_factors", ""),
                            "mda_length": len(sections.get("mda", "")),
                            "risk_factors_length": len(
                                sections.get("risk_factors", "")
                            ),
                            "extraction_success": len(sections) > 0,
                        }

                        all_filings.append(record)

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                continue

        df = pd.DataFrame(all_filings)
        self.logger.info(
            f"Processed filings for {len(df['symbol'].unique())} companies"
        )

        return df

    def save_filings_data(self, filings_df: pd.DataFrame, output_dir: str):
        """
        Save filings data to files

        Args:
            filings_df: DataFrame with filings data
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save main filings data
        filings_path = os.path.join(output_dir, "sec_filings.csv")
        filings_df.to_csv(filings_path, index=False)

        # Save text content separately for analysis
        text_data = []
        for _, row in filings_df.iterrows():
            if row["mda_text"]:
                text_data.append(
                    {
                        "symbol": row["symbol"],
                        "filing_date": row["filing_date"],
                        "form_type": row["form_type"],
                        "section_type": "mda",
                        "text": row["mda_text"],
                    }
                )

            if row["risk_factors_text"]:
                text_data.append(
                    {
                        "symbol": row["symbol"],
                        "filing_date": row["filing_date"],
                        "form_type": row["form_type"],
                        "section_type": "risk_factors",
                        "text": row["risk_factors_text"],
                    }
                )

        text_df = pd.DataFrame(text_data)
        text_path = os.path.join(output_dir, "sec_filings_text.csv")
        text_df.to_csv(text_path, index=False)

        self.logger.info(f"Saved {len(filings_df)} filings to {output_dir}")

    def get_filing_statistics(self, filings_df: pd.DataFrame) -> Dict:
        """
        Get statistics about downloaded filings

        Args:
            filings_df: DataFrame with filings data

        Returns:
            Dictionary with statistics
        """
        if filings_df.empty:
            return {}

        stats = {
            "total_filings": len(filings_df),
            "unique_companies": filings_df["symbol"].nunique(),
            "form_types": filings_df["form_type"].value_counts().to_dict(),
            "extraction_success_rate": filings_df["extraction_success"].mean(),
            "avg_mda_length": filings_df["mda_length"].mean(),
            "avg_risk_factors_length": filings_df["risk_factors_length"].mean(),
            "date_range": {
                "earliest": filings_df["filing_date"].min(),
                "latest": filings_df["filing_date"].max(),
            },
        }

        return stats
