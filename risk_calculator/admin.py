from django.contrib import admin
from .models import Snapshot, PasswordAccess

@admin.register(Snapshot)
class SnapshotAdmin(admin.ModelAdmin):
    list_display = ('id', 'hash', 'timestamp', 'mode', 'user', 'created_at')
    list_filter = ('mode', 'created_at')
    search_fields = ('hash', 'user')
    readonly_fields = ('hash', 'params_json', 'created_at')

@admin.register(PasswordAccess)
class PasswordAccessAdmin(admin.ModelAdmin):
    list_display = ('password', 'is_active', 'created_at', 'updated_at', 'description')
    list_filter = ('is_active', 'created_at')
    search_fields = ('password', 'description')
    readonly_fields = ('created_at', 'updated_at')
    actions = ['deactivate_passwords', 'activate_passwords']

    def deactivate_passwords(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, "Selected passwords have been deactivated.")
    deactivate_passwords.short_description = "Deactivate selected passwords"

    def activate_passwords(self, request, queryset):
        queryset.update(is_active=True)
        self.message_user(request, "Selected passwords have been activated.")
    activate_passwords.short_description = "Activate selected passwords"