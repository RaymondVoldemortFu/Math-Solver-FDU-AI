import dotenv
import os


dotenv.load_dotenv(".env")

api_key = os.getenv("API_KEY")
swanlab_key = os.getenv("SWANLAB_KEY")