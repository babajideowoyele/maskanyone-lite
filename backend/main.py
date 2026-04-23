from fastapi import FastAPI

app = FastAPI()


@app.get("/platform/mode")
def platform_mode():
    return {"mode": "local"}
