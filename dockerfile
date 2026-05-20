# משתמשים ב-Image בסיסי שכבר כולל פייתון
    FROM python:3.9-slim

    # הגדרת תיקיית עבודה בתוך הקונטיינר
    WORKDIR /app

    # התקנת build tools הדרושים עבור Triton compilation
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

    # העתקת רשימת הספריות והתקנתן
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # העתקת שאר הקוד
    COPY . .

    # הפקודה שתרוץ (אפשר לשנות בהתאם לצורך)
    CMD ["python", "train_gpt2.py"]