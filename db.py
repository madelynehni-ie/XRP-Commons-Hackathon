from sqlalchemy import create_engine
from src.config import Config

config = Config()
engine = create_engine(config.database_url, future=True)