from typing import Literal
from pydantic import BaseModel, Field

# Define allowed industry sectors matching the HTML datalist
SectorType = Literal[
    "agriculture and food industry",
    "chemicals and health products",
    "commercial services",
    "communication technologies and media",
    "consumer goods and e-commerce",
    "energy technology",
    "financial and insurance",
    "health and life sciences",
    "hospitality industry",
    "manufacturing industry",
    "real estate and construction",
    "technology and engineering",
    "transport and motorization",
]

class FinancialSchemas(BaseModel):
    """Structured financial metrics extracted from unstructured documents."""

    net_income: float = Field(
        description="The net income or net profit of the company."
    )
    
    net_cash_flow: float = Field(
        description="Total net cash flow from operating, investing, "
        "and financing activities."
    )

    roe: float = Field(
        description="Return on Equity. If not explicitly stated in " \
        "the document, calculate it as (Net Income / Total Equity). " \
        "Express strictly as a decimal."
    )

    roa: float = Field(
        description="Return on Assets. If not explicitly stated in the " \
        "document, calculate it as (Net Income / Total Assets). Express " \
        "strictly as a decimal."
    )

    ebitda: float = Field(
        description="Earnings Before Interest, Taxes, Depreciation, and " \
        "Amortization. If not explicitly stated, calculate as Operating Income " \
        "+ Depreciation + Amortization."
    )

    sector: SectorType = Field(
        description="Industry sector. Must be selected strictly from the allowed " \
        "set of categories."
    )

    cumulation: int = Field(
        description=(
            "Binary indicator for report period length: 0 if the financial data "
            "covers a single quarter (3 months, no cumulation). 1 if the data is "
            "cumulated over a longer period, such as half-year (6M), 9 months, or full year (12M)."
        )
    )