import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

class APIFootballClient:
    BASE_URL = "https://v3.football.api-sports.io"
    
    def __init__(self):
        self.api_key = os.getenv('API_FOOTBALL_KEY')
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY not found in .env file")
        
        self.headers = {
            'x-apisports-key': self.api_key
        }
    
    def _get(self, endpoint, params=None):
        url = f"{self.BASE_URL}/{endpoint}"

        response = requests.get(url, headers=self.headers, params=params)

        try:
            data = response.json()
        except Exception:
            raise Exception(f"Invalid JSON response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {data}")

        # API-specific error handling
        if "errors" in data and data["errors"]:
            raise Exception(f"API returned errors: {data['errors']}")

        time.sleep(1)
        return data
    
    def get_premier_league_id(self):
        """Premier League = 39"""
        return 39
    
    def get_champions_league_id(self):
        """Champions League = 2"""
        return 2
    
    def get_arsenal_team_id(self):
        """Arsenal = 42"""
        return 42
    
    def get_current_season_fixtures(self, league_id, team_id, season=2025):
        """Get all fixtures for a team in a league/season"""
        params = {
            'league': league_id,
            'season': season,
            'team': team_id
        }
        data = self._get('fixtures', params)
        return data['response']
    
    def get_league_standings(self, league_id, season=2025):
        """Get current league table"""
        params = {
            'league': league_id,
            'season': season
        }
        data = self._get('standings', params)
        return data['response'][0]['league']['standings'][0]
    
    def get_team_statistics(self, league_id, team_id, season=2025):
        """Get team stats (form, goals, etc)"""
        params = {
            'league': league_id,
            'season': season,
            'team': team_id
        }
        data = self._get('teams/statistics', params)
        return data['response']