# ibkr_trading_bot/core/utils/notifier_email.py
import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

# vezme .env z kořene projektu (kde ho máte)
load_dotenv()

def _send_mail(to_addr: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    if not (host and port and user and pwd and to_addr):
        return False
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=20) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)
    return True

def notify_flip(symbol: str, timeframe: str, old_sig: str, new_sig: str, price: float, ts_str: str) -> bool:
    """
    Odešle krátký e-mail při změně LONG<->SHORT.
    Vrací True, pokud se podařilo odeslat aspoň jednomu příjemci.
    """
    subj = f"[TradeAlert] {symbol} {timeframe}: {old_sig} -> {new_sig}"
    body = f"{symbol} {timeframe}: {old_sig} -> {new_sig} @ {price:.2f} [{ts_str}]"
    recipients = [x.strip() for x in (os.getenv("ALERT_EMAIL_TO") or "").split(",") if x.strip()]
    ok_any = False
    for to in recipients:
        try:
            ok_any = _send_mail(to, subj, body) or ok_any
        except Exception:
            # nechceme shodit GUI kvůli jednomu selhání
            pass
    return ok_any
