from inference import predict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(title = "Heart disease predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000", "http://127.0.0.1:3000"], #frontend
    allow_methods = ["*"],
    allow_headers = ["*"]
)


limiter = Limiter(key_func = get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class Patient(BaseModel):
    age: int = Field(..., example=63)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=1)
    trestbps: int = Field(..., example=3)
    chol: int = Field(..., example=145)
    fbs: int = Field(..., example=233)
    restecg: int = Field(..., example=1)
    thalach: int = Field(..., example=0)
    exang: int = Field(..., example=150)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: int = Field(..., example=0)
    thal: int = Field(..., example=1)


@app.get("/")
def root():
    return {"message": "API :)"}

@app.get("/health")
def heath():
    return {"status":"ok"}

@app.post("/predict")
@limiter.limit("10/minute")
def predict_patient(request:Request, patient: Patient):
    data = patient.model_dump()
    result = predict(data)
    return result