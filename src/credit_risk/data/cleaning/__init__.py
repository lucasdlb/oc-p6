"""Table-specific cleaning modules."""

from credit_risk.data.cleaning.application import ApplicationCleaner
from credit_risk.data.cleaning.bureau import BureauCleaner
from credit_risk.data.cleaning.bureau_balance import BureauBalanceCleaner
from credit_risk.data.cleaning.credit_card import CreditCardCleaner
from credit_risk.data.cleaning.installments import InstallmentsCleaner
from credit_risk.data.cleaning.pos_cash import POSCashCleaner
from credit_risk.data.cleaning.previous_application import PreviousApplicationCleaner
from credit_risk.data.cleaning.registry import CleaningRegistry

__all__ = [
    "CleaningRegistry",
    "ApplicationCleaner",
    "BureauCleaner",
    "BureauBalanceCleaner",
    "PreviousApplicationCleaner",
    "POSCashCleaner",
    "InstallmentsCleaner",
    "CreditCardCleaner",
]
