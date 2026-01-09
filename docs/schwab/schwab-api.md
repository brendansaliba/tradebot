# Information About the Schwab API

Currently, the tradebot is only configured to use the Charles Schwab API.

## Initial Auth Flow

To begin using the Schwab API, the user must first create a [Schwab Developer account](https://developer.schwab.com/). Then, create an App on the developer portal. When created, the Schwab Client will require the `app_key` and `app_secret` from the developer portal. It's recommended to keep those in a `.env` or somewhere else secure.

To begin the initial auth flow and retrieve your bearer (refresh and access tokens), run the following code.

```python
client = SchwabClient(app_key="your_app_key", app_secret="your_app_secret", redirect_uri="your_redirect_uri")
client.authenticate()
```

`redirect_uri` defaults to `https://172.0.0.1`, so it's not necessary to input anything when initializing the client. In terminal, you will see:

```shell
Please authenticate via Schwab at this url: https://api.schwabapi.com/v1/oauth/authorize?client_id=your_app_key&redirect_uri=your_redirect_uri
Paste the returned URL here once authentication is complete:
```

Go to that url and authenticate with your broker account credentials, NOT your developer credentials. This will redirect you to a url similar to `https://127.0.0.1/?code=a_generated_code&session=a_generated_session`. Paste this in the terminal to give to the client.
