
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
    occ_result_list = []
    occ_result_flag = False
    seg_result_list = []
    seg_result_flag = False
    seg_result = ""
    occ_result = ""
    det_mAP = ""
    det_mATE = ""
    det_mASE = ""
    det_mAOE = ""
    det_mAVE = ""
    det_mAAE = ""
    det_NDS = ""
    if result_line_list is not None:
        for idx, result in enumerate(result_line_list):
            if '===> others - IoU = ' in result:
                occ_result_flag = True
            if 'Map Segmentation Result' in result:
                seg_result_flag = True
                
            if occ_result_flag and len(occ_result_list) < 18:
                occ_result_list.append(result.split('===> ')[1])
                
                
            if seg_result_flag and len(seg_result_list) < 35:
                seg_result_list.append(result)
                
            if 'mIoU :' in result:
                seg_result = result.split(':')[1].strip()
            if 'mIoU of 6019 samples: ' in result:
                occ_result = result.split(':')[1].strip()
            if 'mAP: ' in result:
                det_mAP = round(float(result.split(':')[1].strip()), 4)
            if 'mATE: ' in result:
                det_mATE = round(float(result.split(':')[1].strip()), 4)
            if 'mASE: ' in result:
                det_mASE = round(float(result.split(':')[1].strip()), 4)
            if 'mAOE: ' in result:
                det_mAOE = round(float(result.split(':')[1].strip()), 4)
            if 'mAVE: ' in result:
                det_mAVE = round(float(result.split(':')[1].strip()), 4)
            if 'mAAE: ' in result:
                det_mAAE = round(float(result.split(':')[1].strip()), 4)
            if 'NDS: ' in result:
                det_NDS = round(float(result.split(':')[1].strip()), 4)
        
        subject = f"hyundai MTL (in server {server_number}) mAP: {det_mAP}, NDS: {det_NDS}, seg: {seg_result}, occ: {occ_result} - {model}"
        message = f"mAP: {det_mAP}\nmATE: {det_mATE}\nmASE: {det_mASE}\nmAOE: {det_mAOE}\nmAVE: {det_mAVE}\nmAAE: {det_mAAE}\nNDS: {det_NDS}\n"+\
            "\n\n"

        for occ_result in occ_result_list:
            message += occ_result
        for seg_result in seg_result_list:
            message += seg_result
    else:
        subject = f"something model is ended ({model})"
        message = f"something wrong to show the result. see details in server {server_number}."
        
    recipient_emails = ["junghokim@spa.hanyang.ac.kr", 'dylee@spa.hanyang.ac.kr']
    recipient_emails = ["junghokim@spa.hanyang.ac.kr"]
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
    
    
    
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# import sys
# import json

# def send_mail(result_line_list, model, server_number):
#     smtp_server = "smtp.gmail.com"
#     smtp_port = 587
#     server = smtplib.SMTP(smtp_server, smtp_port)
#     server.starttls()

#     sender_email = "kimjh7669@gmail.com"
#     sender_password = "glviltplcmfdvwvh"
#     server.login(sender_email, sender_password)
#     val_dict = {}
    
#     occ_result_list = []
#     occ_result = ""
#     message = ""
#     if result_line_list is not None:
#         for idx, result in enumerate(result_line_list):
#             if "===>" in result and '- IoU' in result:
#                 cla = result.split('===>')[1].split('- IoU')[0].strip()
#                 val = result.split('===>')[1].split('=')[1].strip()
#                 val_dict[cla] = val
#             elif 'mIoU' in result and "===>" in result:
#                 occ_result = result.split(':')[1].strip()
#         subject = f"occ: {occ_result} - {model}"
#         for k, v in val_dict.items():
#             message += f'{k} : {v}\n'
#         message += f"mIoU : {occ_result}"
#     else:
#         subject = f"something model is ended ({model})"
#         message = f"something wrong to show the result. see details in server {server_number}."
        
#     recipient_emails = ["junghokim@spa.hanyang.ac.kr", 'dylee@spa.hanyang.ac.kr']
#     recipient_emails = ["junghokim@spa.hanyang.ac.kr", ]

#     for recipient_email in recipient_emails:
#         msg = MIMEMultipart()
#         msg['From'] = sender_email
#         msg['To'] = recipient_email
#         msg['Subject'] = subject
#         msg.attach(MIMEText(message, 'plain'))
#         text = msg.as_string()
#         server.sendmail(sender_email, recipient_email, text)
        
#     server.quit()
#     print("sent the result")

# if __name__=="__main__":
#     result_txt = sys.argv[1]
#     model = result_txt.split('/')[-2]
    
#     try:
#         server_number = sys.argv[2]
#     except:
#         server_number = ""

#     try:
#         result_line_list = []
#         f = open(result_txt, 'r')
#         while True:
#             line = f.readline()
#             if not line: break
#             result_line_list.append(line)
#     except:
#         result_line_list = None
        
#     send_mail(result_line_list, model, server_number)