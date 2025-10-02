from django.utils import timezone
from django.http import JsonResponse
import jwt
from django.conf import settings
import logging
from .models import PasswordAccess
from datetime import datetime

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
            request.jwt_payload = payload

            # Check if password is still valid
            password_id = payload.get('password_id')
            if not password_id:
                logger.warning("JWT token missing password_id")
                return JsonResponse({'error': 'Authentication required: Invalid token format'}, status=401)

            try:
                password_access = PasswordAccess.objects.get(id=password_id, is_active=True)
            except PasswordAccess.DoesNotExist:
                logger.warning(f"Password with ID {password_id} is inactive or does not exist")
                return JsonResponse({'error': 'Authentication required: Password revoked or inactive'}, status=401)

            # Check if token was issued after the last revocation
            if password_access.last_revoked_at:
                token_iat = datetime.fromtimestamp(payload['iat'], tz=timezone.utc)
                if token_iat < password_access.last_revoked_at:
                    logger.warning(f"Token issued at {token_iat} is older than last revocation at {password_access.last_revoked_at}")
                    return JsonResponse({'error': 'Authentication required: Token invalidated due to password revocation'}, status=401)

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return JsonResponse({'error': 'Authentication required: Token expired'}, status=401)
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return JsonResponse({'error': 'Authentication required: Invalid token'}, status=401)

        return self.get_response(request)