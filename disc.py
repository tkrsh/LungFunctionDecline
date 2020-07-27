from knockknock import slack_sender

webhook_url = "https://discordapp.com/api/webhooks/737146056017575968/UzcX3__SbysSIjQ_O2BGK7ti1h-45yLIV893G6ha23cWntoV9bt3Tv7E1lMcnEu2GZ0m"
@slack_sender(webhook_url=webhook_url, channel="<your_favorite_slack_channel>")
def train_your_nicest_model():
    import time
    time.sleep(1)
    return {'loss': 0.9}  
train_your_nicest_model()