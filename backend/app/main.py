from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
import uuid
import boto3
from sqlalchemy import create_engine, text
from .auth import router as auth_router, get_current_user
from .db import engine

app = FastAPI()
app.include_router(auth_router)

# MinIO / S3 client
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
)

BUCKET = "citysense"

# create bucket if it doesn't exist
try:
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if BUCKET not in existing:
        s3.create_bucket(Bucket=BUCKET)
except Exception as e:
    print("Warning: could not ensure bucket exists:", e)


@app.post("/api/post")
async def create_post(
    description: str = Form(""),
    lat: float = Form(...),
    lng: float = Form(...),
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):

    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image")

    post_id = str(uuid.uuid4())
    file_name = f"{post_id}.jpg"

    try:
        image.file.seek(0)
        s3.upload_fileobj(image.file, BUCKET, file_name, ExtraArgs={"ContentType": image.content_type})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {e}")

    url = f"http://localhost:9000/{BUCKET}/{file_name}"

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO posts(id, user_id, description, category, image_url, lat, lng, status, created_at)
                VALUES(:id, :user_id, :description, :category, :image_url, :lat, :lng, 'PENDING', now())
            """), {
                "id": post_id,
                "user_id": current_user["id"],
                "description": description,
                "category": None,
                "image_url": url,
                "lat": lat,
                "lng": lng
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {"post_id": post_id, "image_url": url, "status": "PENDING"}
