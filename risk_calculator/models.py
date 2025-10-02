from django.db import models

class Snapshot(models.Model):
    hash = models.CharField(max_length=64, unique=True)
    timestamp = models.DateTimeField()
    params_json = models.JSONField()
    mode = models.CharField(
        max_length=20,
        choices=[('public', 'Public'), ('private', 'Private'), ('pro-forma', 'Pro-Forma')]
    )
    user = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['hash', 'timestamp'])]

class PasswordAccess(models.Model):
    password = models.CharField(max_length=128, unique=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    description = models.CharField(max_length=255, blank=True, help_text="Optional description for this password")

    def __str__(self):
        return f"Password ({'Active' if self.is_active else 'Inactive'})"

    class Meta:
        verbose_name = "Access Password"
        verbose_name_plural = "Access Passwords"
        