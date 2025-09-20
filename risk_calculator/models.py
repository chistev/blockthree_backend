from django.db import models


class Snapshot(models.Model):
    hash = models.CharField(max_length=64, unique=True)
    timestamp = models.DateTimeField()
    params_json = models.JSONField()
    mode = models.CharField(
        max_length=20,
        choices=[('public', 'Public'), ('private', 'Private'), ('pro-forma', 'Pro-Forma')]
    )
    user = models.CharField(max_length=100, blank=True)  # Optional, for audit trail
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['hash', 'timestamp'])]