from django.db import models
from django.utils.html import format_html

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    predicted_label = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.predicted_label} ({self.confidence:.2f})" if self.predicted_label else "Unclassified"


    def image_tag(self):
        if self.image:
            return format_html('<img src="{}" width="100" height="100" />', self.image.url)
        return "No Image"

    image_tag.short_description = "Image Preview"
