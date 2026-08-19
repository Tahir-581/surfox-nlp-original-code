import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

log = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.mailgun.org")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "Surfox <event@mg.limeox.com>")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3010")


def _send_email(to: str, subject: str, html_body: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to, msg.as_string())
        log.info("Email sent to %s: %s", to, subject)
        return True
    except Exception as exc:
        log.exception("Failed to send email to %s: %s", to, exc)
        return False


def send_verification_email(to_email: str, name: str, token: str) -> bool:
    verify_url = f"{FRONTEND_URL}/verify-email?token={token}"
    html = f"""
    <div style="font-family: 'Inter', Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #ffffff;">
      <div style="background: linear-gradient(135deg, #FF5C00 0%, #FF8040 100%); padding: 40px 32px; border-radius: 12px 12px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">Surfox</h1>
        <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 14px;">Content Intelligence Platform</p>
      </div>
      <div style="padding: 40px 32px; border: 1px solid #ECEEF2; border-top: none; border-radius: 0 0 12px 12px;">
        <h2 style="color: #1A1D23; font-size: 22px; font-weight: 700; margin: 0 0 12px;">Welcome, {name}! 👋</h2>
        <p style="color: #4B5162; font-size: 15px; line-height: 1.6; margin: 0 0 24px;">
          Thanks for creating your Surfox account. Please verify your email address to get started.
        </p>
        <a href="{verify_url}"
           style="display: inline-block; background: #FF5C00; color: white; text-decoration: none;
                  padding: 14px 32px; border-radius: 8px; font-weight: 700; font-size: 15px;
                  letter-spacing: 0.3px;">
          Verify My Email
        </a>
        <p style="color: #9EA3AF; font-size: 13px; margin: 24px 0 0; line-height: 1.5;">
          This link expires in 24 hours. If you didn't create this account, you can safely ignore this email.
        </p>
        <hr style="border: none; border-top: 1px solid #ECEEF2; margin: 24px 0;" />
        <p style="color: #9EA3AF; font-size: 12px; margin: 0;">
          Or copy this link: <a href="{verify_url}" style="color: #FF5C00;">{verify_url}</a>
        </p>
      </div>
    </div>
    """
    return _send_email(to_email, "Verify your Surfox account", html)


def send_reset_password_email(to_email: str, name: str, token: str) -> bool:
    reset_url = f"{FRONTEND_URL}/reset-password?token={token}"
    html = f"""
    <div style="font-family: 'Inter', Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #ffffff;">
      <div style="background: linear-gradient(135deg, #FF5C00 0%, #FF8040 100%); padding: 40px 32px; border-radius: 12px 12px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 28px; font-weight: 800;">Surfox</h1>
        <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 14px;">Content Intelligence Platform</p>
      </div>
      <div style="padding: 40px 32px; border: 1px solid #ECEEF2; border-top: none; border-radius: 0 0 12px 12px;">
        <h2 style="color: #1A1D23; font-size: 22px; font-weight: 700; margin: 0 0 12px;">Reset Your Password</h2>
        <p style="color: #4B5162; font-size: 15px; line-height: 1.6; margin: 0 0 8px;">Hi {name},</p>
        <p style="color: #4B5162; font-size: 15px; line-height: 1.6; margin: 0 0 24px;">
          We received a request to reset your password. Click the button below to create a new one.
        </p>
        <a href="{reset_url}"
           style="display: inline-block; background: #FF5C00; color: white; text-decoration: none;
                  padding: 14px 32px; border-radius: 8px; font-weight: 700; font-size: 15px;">
          Reset Password
        </a>
        <p style="color: #9EA3AF; font-size: 13px; margin: 24px 0 0; line-height: 1.5;">
          This link expires in 1 hour. If you didn't request a password reset, please ignore this email.
        </p>
        <hr style="border: none; border-top: 1px solid #ECEEF2; margin: 24px 0;" />
        <p style="color: #9EA3AF; font-size: 12px; margin: 0;">
          Or copy this link: <a href="{reset_url}" style="color: #FF5C00;">{reset_url}</a>
        </p>
      </div>
    </div>
    """
    return _send_email(to_email, "Reset your Surfox password", html)
