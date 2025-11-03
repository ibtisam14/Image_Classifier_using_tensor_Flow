from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    predicted_label = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.predicted_label or "Unclassified"
