"""Assign a category to each transaction from its narration.

Rule-based on purpose: categories feed numbers the user will act on, so the
same narration must always land in the same category, and every decision must
be explainable ("matched 'swiggy'"). A model would be neither deterministic
nor cheap at 50,000 rows per upload.

Rules are *data*: an ordered list of (category, direction, keywords). The
first match wins, so specific rules go before broad ones ("swiggy instamart"
is groceries before "swiggy" is food). Keywords match whole words in the
lower-cased narration, so "uber" does not match "cucumber".

The :class:`Categoriser` protocol lets a smarter implementation (a learned
model, per-user corrections) replace this one without touching callers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from money_agent.domain.models import Category, Direction


class Categoriser(Protocol):
    """Assigns categories."""

    def categorise(self, description: str, direction: Direction) -> Category:
        """Pick a category.

        Args:
            description: Bank narration.
            direction: Money in or out.

        Returns:
            The category (never fails: unknown narrations get a fallback).
        """
        ...


@dataclass(frozen=True, slots=True)
class Rule:
    """One categorisation rule.

    Attributes:
        category: Category assigned on a match.
        keywords: Whole words or phrases, lower case.
        direction: Only match this direction (``None``: either).
    """

    category: Category
    keywords: tuple[str, ...]
    direction: Direction | None = None


DEBIT, CREDIT = Direction.DEBIT, Direction.CREDIT

# Order matters: specific before general.
DEFAULT_RULES: tuple[Rule, ...] = (
    # Money in
    Rule(Category.SALARY, ("salary", "sal credit", "payroll", "salary credit"), CREDIT),
    Rule(Category.REFUND, ("refund", "reversal", "cashback", "chargeback"), CREDIT),
    Rule(Category.OTHER_INCOME, ("interest", "int pd", "dividend", "int credit"), CREDIT),
    # Bank charges before everything that mentions a card or account
    Rule(Category.FEES, ("charges", "chrg", "annual fee", "penalty", "late fee", "gst on",
                         "sms alert", "min bal"), DEBIT),
    Rule(Category.LOAN_EMI, ("emi", "loan", "nach emi", "bajaj finance", "home loan"), DEBIT),
    Rule(Category.INVESTMENTS, ("sip", "mutual fund", "mf", "zerodha", "groww", "kuvera",
                                "coin by zerodha", "nps", "ppf", "upstox", "indmoney"), DEBIT),
    Rule(Category.INSURANCE, ("insurance", "lic", "premium", "policybazaar", "acko",
                              "hdfc ergo", "star health"), DEBIT),
    Rule(Category.RENT, ("rent", "nobroker", "house rent", "landlord"), DEBIT),
    Rule(Category.GROCERIES, ("instamart", "blinkit", "zepto", "bigbasket", "dmart",
                              "grofers", "jiomart", "more retail", "grocery", "kirana",
                              "reliance fresh"), DEBIT),
    Rule(Category.FOOD_DINING, ("swiggy", "zomato", "restaurant", "cafe", "dominos",
                                "mcdonalds", "kfc", "starbucks", "pizza", "eatsure",
                                "food"), DEBIT),
    Rule(Category.FUEL, ("petrol", "fuel", "hpcl", "bpcl", "indian oil", "iocl",
                         "shell"), DEBIT),
    Rule(Category.TRANSPORT, ("uber", "ola", "rapido", "metro", "irctc", "redbus",
                              "fastag", "parking", "namma yatri"), DEBIT),
    Rule(Category.TRAVEL, ("makemytrip", "goibibo", "indigo", "air india", "vistara",
                           "akasa", "cleartrip", "oyo", "airbnb", "hotel"), DEBIT),
    Rule(Category.SUBSCRIPTIONS, ("netflix", "spotify", "hotstar", "prime video",
                                  "amazon prime", "youtube premium", "jiocinema", "sonyliv",
                                  "apple com bill", "google play", "chatgpt",
                                  "subscription"), DEBIT),
    Rule(Category.UTILITIES, ("electricity", "bescom", "msedcl", "tata power", "broadband",
                              "airtel", "jio", "vi prepaid", "recharge", "postpaid", "gas",
                              "water bill", "act fibernet", "bill payment"), DEBIT),
    Rule(Category.SHOPPING, ("amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho",
                             "croma", "reliance digital", "decathlon", "ikea"), DEBIT),
    Rule(Category.ENTERTAINMENT, ("bookmyshow", "pvr", "inox", "district", "steam",
                                  "playstation"), DEBIT),
    Rule(Category.HEALTH, ("pharmacy", "apollo", "medplus", "1mg", "pharmeasy", "hospital",
                           "clinic", "diagnostic", "practo", "netmeds"), DEBIT),
    Rule(Category.EDUCATION, ("school", "college", "university", "tuition", "udemy",
                              "coursera", "byjus", "unacademy", "fees"), DEBIT),
    Rule(Category.CASH, ("atm", "cash wdl", "cash withdrawal", "atw", "nwd"), DEBIT),
    Rule(Category.TRANSFERS, ("neft", "imps", "rtgs", "upi", "transfer", "self"), None),
)


def _pattern(keyword: str) -> re.Pattern[str]:
    words = (re.escape(word) for word in keyword.split())
    return re.compile(r"(?<![a-z0-9])" + r"[\s\W_]+".join(words) + r"(?![a-z0-9])")


class RuleCategoriser:
    """First-match keyword rules."""

    def __init__(self, rules: Sequence[Rule] = DEFAULT_RULES) -> None:
        """Compile the rules.

        Args:
            rules: Ordered rules (first match wins).

        Raises:
            ValueError: If a rule has no keywords.
        """
        if any(not rule.keywords for rule in rules):
            raise ValueError("every rule needs at least one keyword")
        self._rules = [(rule, [_pattern(k) for k in rule.keywords]) for rule in rules]

    def categorise(self, description: str, direction: Direction) -> Category:
        """Pick a category (see :class:`Categoriser`).

        Args:
            description: Bank narration.
            direction: Money in or out.

        Returns:
            The first matching rule's category, else ``other_income`` for money
            in and ``other`` for money out.
        """
        text = description.lower()
        for rule, patterns in self._rules:
            if rule.direction not in (None, direction):
                continue
            if any(p.search(text) for p in patterns):
                return rule.category
        return Category.OTHER_INCOME if direction is Direction.CREDIT else Category.OTHER
