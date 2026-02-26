from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class UserProfile(BaseModel):
    id: str
    age: int
    gender: str
    medical_constraints: List[str]
    preferences: List[str]
    biometrics: dict

# Mock database
DB = {
    "user1": UserProfile(
        id="user1",
        age=32,
        gender="Unknown",
        medical_constraints=["mild knee pain"],
        preferences=["vegetarian", "lose weight"],
        biometrics={}
    )
}

@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    if user_id in DB:
        return DB[user_id]
    raise HTTPException(status_code=404, detail="User not found")
