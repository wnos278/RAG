import os
from dotenv import load_dotenv

# Load các biến môi trường từ file .env
load_dotenv()


OPENAI_KEY=os.getenv('OPENAI_KEY')
