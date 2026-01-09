import os
from pathlib import Path
from dotenv import load_dotenv, set_key
import base64
import requests
from datetime import datetime, timezone
from dateutil import parser
import json

class SchwabClient():
    def __init__(self, app_key: str = None, app_secret: str = None, redirect_uri: str = "https://172.0.0.1", refresh_token: str = None) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._redirect_uri = redirect_uri
        self._base_url = "https://api.schwabapi.com/trader/v1"
        self._date = self._get_date()

        # load tokens from storage
        self._refresh_token = refresh_token
        self._access_token = None
        self._id_token = None

        if not self._refresh_token:
            self.authenticate()
        elif not self._access_token:
            self.refresh_tokens()

        self._account_number, self._account_hash_value = self._get_account_number()
        self.session_hours = self.get_schwab_equity_session_hours()
        self._accounts = self._get_accounts()


    def _construct_auth_url(self) -> str:
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={self._app_key}&redirect_uri={self._redirect_uri}"
        return auth_url

    def _construct_auth_package(self, returned_url: str = None) -> dict[str, str]:
        response_code = f"{returned_url[returned_url.index('code=') + 5: returned_url.index('%40')]}@"

        credentials = f"{self._app_key}:{self._app_secret}"
        base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode(
            "utf-8"
        )

        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        payload = {
            "grant_type": "authorization_code",
            "code": response_code,
            "redirect_uri": self._redirect_uri,
        }

        return {"headers": headers, "payload": payload}
    
    def _retrieve_tokens(self, package):
        headers = package["headers"]
        payload = package["payload"]

        init_token_response = requests.post(
            url="https://api.schwabapi.com/v1/oauth/token",
            headers=headers,
            data=payload,
        )

        tokens = init_token_response.json()

        self._refresh_token = tokens["refresh_token"]
        self._access_token = tokens["access_token"]
        self._id_token = tokens["id_token"]

        return tokens
    
    def _get_account_number(self) -> tuple[str, str]:
        headers = {"Authorization": f"Bearer {self._access_token}"}

        res = requests.get(
            self._base_url + f"/accounts/accountNumbers", headers=headers
        )

        res_l = res.json()
        red_d = res_l[0]
        account_number = red_d["accountNumber"]
        account_hash_value = red_d["hashValue"]
        
        return [account_number, account_hash_value]
    
    def _get_date(self) -> str:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        return date
    
    def get_datetime(self) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S%z")

    @property
    def account_number(self):
        return self._account_number

    def authenticate(self) -> None:
        auth_url = self._construct_auth_url()
        print("Authenticate via Schwab at this url:", auth_url)
        print("Paste the returned URL here once authentication is complete:")
        returned_url = input()
        package = self._construct_auth_package(returned_url=returned_url)
        tokens = self._retrieve_tokens(package=package)

        self._refresh_token = tokens["refresh_token"]
        self._access_token = tokens["access_token"]
        self._id_token = tokens["id_token"]

        print("Authentication complete.")

    def refresh_tokens(self):
        if not self._refresh_token:
            self.authenticate()
            return
        
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }
        headers = {
            "Authorization": f'Basic {base64.b64encode(f"{self._app_key}:{self._app_secret}".encode()).decode()}',
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = requests.post(
            url="https://api.schwabapi.com/v1/oauth/token",
            headers=headers,
            data=payload,
        )

        if response.status_code == 200:
            print("Retrieved new tokens successfully using refresh token.")
        else:
            raise ConnectionError(
                f"Error refreshing access token: {response.text}"
            )

        tokens = response.json()

        self._refresh_token = tokens["refresh_token"]
        self._access_token = tokens["access_token"]
        self._id_token = tokens["id_token"]
        print("Tokens refreshed.")

    def get_schwab_equity_session_hours(self) -> dict:
        url = "https://api.schwab.com/marketdata/v1/markets"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json"
        }
        params = {}
        params["markets"] = "equity"
        params["date"] = self._date
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        market_hours = response.json()

        if "equity" in market_hours:
            session_hours = market_hours["equity"]["EQ"]["sessionHours"]
            
        return session_hours
    
    @property
    def pre_market(self) -> bool:
        now = self.get_datetime()
        now = parser.isoparse(now.replace('+0000', '+00:00'))
    
        start = parser.isoparse(self.session_hours["preMarket"][0]["start"])
        end = parser.isoparse(self.session_hours["preMarket"][0]["end"])

        return start <= now < end
    
    @property
    def regular_market(self) -> bool:
        now = self.get_datetime()
        now = parser.isoparse(now.replace('+0000', '+00:00'))
    
        start = parser.isoparse(self.session_hours["regularMarket"][0]["start"])
        end = parser.isoparse(self.session_hours["regularMarket"][0]["end"])

        return start <= now < end
    
    @property
    def post_market(self) -> bool:
        now = self.get_datetime()
        now = parser.isoparse(now.replace('+0000', '+00:00'))
    
        start = parser.isoparse(self.session_hours["postMarket"][0]["start"])
        end = parser.isoparse(self.session_hours["postMarket"][0]["end"])

        return start <= now < end
    
    def _get_accounts(self):
        url = "https://api.schwab.com/trader/v1/accounts"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json"
        }
        params = {}
        params["fields"] = "positions"
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        accounts = response.json()

        self._accounts = accounts