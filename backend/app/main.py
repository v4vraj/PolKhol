from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status,Request
import uuid
import boto3
from boto3.session import Config as BotoConfig
from botocore.exceptions import ClientError
from sqlalchemy import text
from .auth import router as auth_router, get_current_user
from .db import engine
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI()
app.include_router(auth_router)

# MinIO / S3 client
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
    config=BotoConfig(signature_version="s3v4"),
)

BUCKET = "citysense"

# create bucket if it doesn't exist
try:
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if BUCKET not in existing:
        s3.create_bucket(Bucket=BUCKET)
except Exception as e:
    print("Warning: could not ensure bucket exists:", e)


class UploadRequest(BaseModel):
    filename: str
    content_type: Optional[str] = "image/jpeg"

class UploadResponse(BaseModel):
    url: str
    fields: dict
    object_url: str

class PostCreate(BaseModel):
    description: str
    lat: float
    lng: float
    image_url: str


@app.post("/api/post_file")
async def create_post_file(
    description: str = Form(""),
    lat: float = Form(...),
    lng: float = Form(...),
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Backwards-compatible endpoint: client uploads file to server (not recommended for prod).
    """
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image")

    post_id = str(uuid.uuid4())
    ext = image.filename.split(".")[-1] if "." in image.filename else "jpg"
    object_key = f"{post_id}.{ext}"

    try:
        image.file.seek(0)
        s3.upload_fileobj(image.file, BUCKET, object_key, ExtraArgs={"ContentType": image.content_type})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {e}")

    object_url = f"{s3.meta.endpoint_url}/{BUCKET}/{object_key}"

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
                "image_url": object_url,
                "lat": lat,
                "lng": lng
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {"post_id": post_id, "image_url": object_url, "status": "PENDING"}

@app.post("/api/upload-url", response_model=UploadResponse)
def create_upload_url(req: UploadRequest, current_user: dict = Depends(get_current_user)):
    """
    Returns a presigned POST (url + fields) for direct client -> MinIO upload.
    Client should POST a multipart/form-data with the returned fields plus the file.
    """
    # derive extension and stable object key
    ext = req.filename.split(".")[-1] if "." in req.filename else "jpg"
    object_key = f"{uuid.uuid4()}.{ext}"

    try:
        presigned = s3.generate_presigned_post(
            Bucket=BUCKET,
            Key=object_key,
            Fields={"Content-Type": req.content_type},
            Conditions=[
                {"Content-Type": req.content_type},
                ["content-length-range", 1, 10 * 1024 * 1024]  # allow up to 10 MB (adjust as needed)
            ],
            ExpiresIn=900  # seconds (15 minutes)
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create upload url: {e}")

    object_url = f"{s3.meta.endpoint_url}/{BUCKET}/{object_key}"
    return UploadResponse(url=presigned["url"], fields=presigned["fields"], object_url=object_url)

@app.post("/api/post")
def create_post(payload: PostCreate, current_user: dict = Depends(get_current_user)):
    """
    Client calls this AFTER they have uploaded the file directly to MinIO
    using the presigned POST /api/upload-url. This endpoint simply registers the post.
    """
    # basic validation - ensure image_url belongs to our BUCKET domain (optional)
    if not payload.image_url.startswith(str(s3.meta.endpoint_url)) and BUCKET not in payload.image_url:
        # allow it but warn — you may want to enforce only MinIO objects
        print("[warning] image_url does not appear to be the local MinIO URL")

    post_id = str(uuid.uuid4())
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO posts(id, user_id, description, category, image_url, lat, lng, status, created_at)
                VALUES(:id, :user_id, :description, :category, :image_url, :lat, :lng, 'PENDING', now())
            """), {
                "id": post_id,
                "user_id": current_user["id"],
                "description": payload.description,
                "category": None,
                "image_url": payload.image_url,
                "lat": payload.lat,
                "lng": payload.lng
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    # TODO: trigger Kestra perception workflow here (call Kestra API with post_id + image_url)

    return {"post_id": post_id, "image_url": payload.image_url, "status": "PENDING"}


@app.post("/api/perception_callback")
async def perception_callback(request: Request):
    payload = await request.json()
    post_id = payload.get("post_id")
    result = payload.get("result") or {}

    if not post_id or not result:
        raise HTTPException(status_code=400, detail="post_id and result are required")

    category = result.get("category")
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    embedding = result.get("embedding")  # list of floats
    model_used = result.get("model", None)

    severity_score = confidence
    authenticity_score = 0.5

    with engine.begin() as conn:
        # update posts
        conn.execute(text("""
            UPDATE posts
            SET category = :category,
                severity_score = :severity,
                authenticity_score = :auth,
                status = 'ANALYZED',
                updated_at = now()
            WHERE id = :post_id
        """), {"category": category, "severity": severity_score, "auth": authenticity_score, "post_id": post_id})

        # insert embedding into pgvector table 'embeddings' if available
        if embedding:
            try:
                # build a safe numeric-only string like "[0.1,0.2,...]"
                emb_str = "[" + ",".join(map(str, embedding)) + "]"
                insert_stmt = text("""
                    INSERT INTO embeddings (post_id, embedding, created_at)
                    VALUES (:post_id, (:embedding)::vector, now())
                    RETURNING id
                """)
                r = conn.execute(insert_stmt, {"post_id": post_id, "embedding": emb_str})
                try:
                    inserted_row = r.fetchone()
                    embedding_inserted = 1 if inserted_row else (r.rowcount or 1)
                except Exception:
                    embedding_inserted = r.rowcount or 1
            except Exception as e_emb:
                # fallback to embeddings_json if needed (your existing fallback)
                try:
                    conn.execute(text("""
                        INSERT INTO embeddings_json (post_id, embedding_json, model)
                        VALUES (:post_id, :embedding_json, :model)
                    """), {"post_id": post_id, "embedding_json": json.dumps(embedding), "model": model_used})
                    embedding_inserted = 1
                except Exception as e2:
                    print("[perception_callback] embed insert failed:", e_emb, "fallback failed:", e2)
                    raise HTTPException(status_code=500, detail=f"Embedding insert error: {e_emb}")

    return {"status": "ok", "post_id": post_id, "category": category, "confidence": confidence}