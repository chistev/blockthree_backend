from django.http import JsonResponse
from django.urls import resolve

class PasswordProtectionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip middleware for admin panel and specific endpoints
        exempt_urls = ['/api/login/', '/admin/', '/static/', '/media/']
        if any(request.path.startswith(url) for url in exempt_urls):
            return self.get_response(request)

        # Check if user is authenticated
        if not request.session.get('is_authenticated', False):
            return JsonResponse({'error': 'Authentication required'}, status=401)

        return self.get_response(request)