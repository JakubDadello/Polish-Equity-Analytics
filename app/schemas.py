from pydantic import BaseModel

class Model_Request (BaseModel):
   # --- Financial indicators ---
   net_income: float
   net_cash_flow: float 
   roe: float
   roa: float
   ebitda: float

   # --- Categorical and technical features ---
   sector: str
   cumulation: int  
