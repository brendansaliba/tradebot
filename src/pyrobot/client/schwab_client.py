import os
from pathlib import Path
from dotenv import load_dotenv, set_key
import base64
import requests
from datetime import datetime, timezone
from dateutil import parser
import json
from typing import Union

class SchwabClient():
    def __init__(self, app_key: str = None, app_secret: str = None, redirect_uri: str = "https://127.0.0.1", refresh_token: str = None, access_token: str = None, id_token: str = None) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._redirect_uri = redirect_uri
        self._trader_url = "https://api.schwabapi.com/trader/v1"
        self._marketdata_url = "https://api.schwab.com/marketdata/v1"
        self._date = self._get_date()
        self._market_open_day = False 

        # tokens
        self._refresh_token = refresh_token
        self._access_token = access_token
        self._id_token = id_token

        if not self._refresh_token:
            print("No refresh token found. Begin auth flow")
            self.authenticate()
        elif not self._access_token:
            print("No access token found. Begin token refesh flow")
            self.refresh_tokens()

        self._account_list = self._get_account_list()
        self._account_number, self._account_hash_value = self._get_account_number()
        self.session_hours = self.get_equity_session_hours()
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
    
    def _get_account_list(self):
        headers = {"Authorization": f"Bearer {self._access_token}"}

        res = requests.get(
            self._trader_url + f"/accounts/accountNumbers", headers=headers
        )

        if res.status_code == 200:
            print("Retrieved account list.")
            account_list = res.json()
            return account_list
        else:
            print('There was an issue getting accounts')
            return None
        
    
    def _get_account_number(self) -> tuple[str, str]:
        headers = {"Authorization": f"Bearer {self._access_token}"}

        res = requests.get(
            self._trader_url + f"/accounts/accountNumbers", headers=headers
        )

        if res.status_code == 200:
            print("Retrieved account list.")
            account_list = res.json()
            red_d = account_list[0]
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
        returned_url = input("Paste the returned URL here once authentication is complete:")

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
            tokens = response.json()
            self._refresh_token = tokens["refresh_token"]
            self._access_token = tokens["access_token"]
            self._id_token = tokens["id_token"]
            print("Tokens refreshed.")
        elif response.status_code == 401:
            print("Unauthorized. Restart auth flow to get new refresh token.")
            self.authenticate()
            return
        else:
            self.authenticate()
            return


    def get_equity_session_hours(self, date: str = None) -> Union[dict, None]:
        url = f"{self._marketdata_url}/markets/equity"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json"
        }
        params = {}
        params["markets"] = "equity"
        if date is not None:
            params["date"] = date
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        market_hours = response.json()

        if "equity" in market_hours:
            if "EQ" in market_hours["equity"]:
                session_hours = market_hours["equity"]["EQ"]["sessionHours"]
                self._market_open_day = True
                return session_hours
            elif "equity" in market_hours["equity"]:
                if not market_hours["equity"]["equity"]["isOpen"]:
                    self._market_open_day = False
                    return None
    
    @property
    def pre_market(self) -> Union[bool, None]:
        if self._market_open_day:
            now = self.get_datetime()
            now = parser.isoparse(now.replace('+0000', '+00:00'))
            start = parser.isoparse(self.session_hours["preMarket"][0]["start"])
            end = parser.isoparse(self.session_hours["preMarket"][0]["end"])
            return start <= now < end
        else:
            return None
    
    @property
    def regular_market(self) -> Union[bool, None]:
        if self._market_open_day:
            now = self.get_datetime()
            now = parser.isoparse(now.replace('+0000', '+00:00'))
            start = parser.isoparse(self.session_hours["regularMarket"][0]["start"])
            end = parser.isoparse(self.session_hours["regularMarket"][0]["end"])
            return start <= now < end
        else:
            return None
    
    @property
    def post_market(self) -> Union[bool, None]:
        if self._market_open_day:
            now = self.get_datetime()
            now = parser.isoparse(now.replace('+0000', '+00:00'))
            start = parser.isoparse(self.session_hours["postMarket"][0]["start"])
            end = parser.isoparse(self.session_hours["postMarket"][0]["end"])
            return start <= now < end
        else:
            return None
    
    def _get_accounts(self):
        url = f"{self._trader_url}/accounts"
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

    def get_historical_prices(self, symbol: str = None, period_type: str = "", period: int = None, frequency_type: str = "", frequency: int = None, start: str = None, end: str = None, extended_hours: bool = True, previous_close: bool = True):
        url = f"{self._marketdata_url}/pricehistory"
        params = {
            "symbol": symbol.upper(),
            "periodType": period_type,
            "needExtendedHoursData": str(extended_hours).lower(),
            "needPreviousClose": str(previous_close).lower()
        }

        if period:
            params["period"] = period
        if frequency_type:
            params["frequencyType"] = frequency_type
        if period:
            params["frequency"] = frequency
        if start:
            params["startDate"] = start
        if end:
            params["endDate"] = end

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json"
        }

        try:
            r = requests.get(url, headers=headers, params=params, timeout=12)
            
            if r.status_code == 401:
                raise ValueError("401 Unauthorized → most likely expired or invalid access token")
            if r.status_code == 403:
                raise PermissionError("403 Forbidden → app probably not approved for Market Data")
            if r.status_code == 429:
                raise RuntimeError("429 Rate limit hit")
                
            r.raise_for_status()
            print(json.dumps(r.json()))
            return r.json()
        
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP {r.status_code} - {r.text}") from e
        except Exception as e:
            raise RuntimeError(f"Request failed: {str(e)}") from e