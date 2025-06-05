from fastapi import FastAPI

app = FastAPI(title="MoFe API", description="MoFe Backend API Service")


# 内存中存储数据
items_db = []


@app.get("/")
async def root():
    return {"message": "Welcome to MoFe API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
