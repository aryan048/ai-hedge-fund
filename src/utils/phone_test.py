import smtplib
from email.message import EmailMessage

YOUR_EMAIL = "aryan.hazra04@gmail.com"
YOUR_PASSWORD = "vqrf irbp ewbx dszg"
TO_PHONE_NUMBER = "6787070660@tmomail.net"  # Change to match your carrier

msg = EmailMessage()
msg.set_content("Hello! This is a test text from Python.")
msg["Subject"] = "SMS Notification"
msg["From"] = YOUR_EMAIL
msg["To"] = TO_PHONE_NUMBER

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(YOUR_EMAIL, YOUR_PASSWORD)
    server.send_message(msg)

print("Message sent!")