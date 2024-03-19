
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sys
import json

def send_mail(result_line_list, model, server_number):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()

    sender_email = "kimjh7669@gmail.com"
    sender_password = "glviltplcmfdvwvh"
    server.login(sender_email, sender_password)
    val_dict = {}
    
    occ_result_list = []
    occ_result = ""
    message = ""
    if result_line_list is not None:
        for idx, result in enumerate(result_line_list):
            if "===>" in result and '- IoU' in result:
                cla = result.split('===>')[1].split('- IoU')[0].strip()
                val = result.split('===>')[1].split('=')[1].strip()
                val_dict[cla] = val
            elif 'mIoU' in result and "===>" in result:
                occ_result = result.split(':')[1].strip()
        subject = f"occ: {occ_result} - {model}"
        for k, v in val_dict.items():
            message += f'{k} : {v}\n'
        message += f"mIoU : {occ_result}"
    else:
        subject = f"something model is ended ({model})"
        message = f"something wrong to show the result. see details in server {server_number}."
        
    recipient_emails = ["junghokim@spa.hanyang.ac.kr", 'dylee@spa.hanyang.ac.kr']
    recipient_emails = ["junghokim@spa.hanyang.ac.kr", ]

    for recipient_email in recipient_emails:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        
    server.quit()
    print("sent the result")

if __name__=="__main__":
    result_txt = sys.argv[1]
    model = result_txt.split('/')[-2]
    
    try:
        server_number = sys.argv[2]
    except:
        server_number = ""

    try:
        result_line_list = []
        f = open(result_txt, 'r')
        while True:
            line = f.readline()
            if not line: break
            result_line_list.append(line)
    except:
        result_line_list = None
        
    send_mail(result_line_list, model, server_number)