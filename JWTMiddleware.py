from fastapi import HTTPException, Request
import jwt

# Define the JWT middleware function
async def jwt_middleware(request: Request, call_next):

    authorization = request.headers.get('Authorization')

    # Check if the Authorization header is present
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    # Check if the Authorization header is formatted correctly
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    # Extract the JWT token from the Authorization header
    token = authorization.split(" ")[1]

    try:
        # Decode the JWT token using a secret key and algorithm
        payload = jwt.decode(token, "dunkdink", algorithms=["HS256"])

        # Add the decoded payload to the request object as a new attribute
        if not payload['sub']:
            raise HTTPException(status_code=401, detail="Invalid token")

    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")

    return await call_next(request)