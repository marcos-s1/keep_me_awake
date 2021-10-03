from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "AC9017aad372629411b5bd7a2a499f02b9"
# Your Auth Token from twilio.com/console
auth_token = "5adef537f4f895a07b5f04b5c5497ef2"

# client credentials are read from TWILIO_ACCOUNT_SID and AUTH_TOKEN
client = Client(account_sid, auth_token)

message = client.messages.create(
        to="+5591991260029",
        from_="+16062682843",
        body="ISSO É UM TESTE! SISTEMA TOTALMENTE FUNCIONAL")

print(message.sid)
