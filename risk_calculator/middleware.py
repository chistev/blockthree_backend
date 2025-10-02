from django.http import JsonResponse
import jwt
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class PasswordProtectionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip middleware for admin panel and specific endpoints
        exempt_urls = ['/api/login/', '/admin/', '/static/', '/media/']
        if any(request.path.startswith(url) for url in exempt_urls):
            return self.get_response(request)

        # Check for JWT in Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header")
            return JsonResponse({'error': 'Authentication required: Missing or invalid token'}, status=401)

        token = auth_header.split(' ')[1]
        try:
            # Decode JWT
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=['HS256'])
            # Optionally, you can attach user info to the request
            request.jwt_payload = payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return JsonResponse({'error': 'Authentication required: Token expired'}, status=401)
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return JsonResponse({'error': 'Authentication required: Invalid token'}, status=401)

        return self.get_response(request)