
import pandas as pd

def sendMessage():

	import requests

	url = "https://www.fast2sms.com/dev/bulk"

	payload = "sender_id=FSTSMS&message=The%20model%20have%20been%20trained%20successfully&language=english&route=p&numbers=8126102904"
	headers = {
	    'authorization': "OuX3BnsANZWIvpQzT70ikgD4ERwehbyHdV2Ff6MmSL51jCU8KoHnlJ3aR0FVTeK6buhgvmy9dkL7zqYj",
	    'Content-Type': "application/x-www-form-urlencoded",
	    'Cache-Control': "no-cache",
	    }

	response = requests.request("POST", url, data=payload, headers=headers)

	print(response.text)
