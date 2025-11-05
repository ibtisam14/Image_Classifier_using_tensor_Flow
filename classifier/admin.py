from django.contrib import admin
from .models import UploadedImage

@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ("id", "image_tag", "predicted_label", "confidence")
    list_filter = ("predicted_label",)  
    search_fields = ("predicted_label",)
