from django.db import models

# Create your models here.
class Prediction(models.Model):
    frame = models.CharField(max_length=255)
    x = models.FloatField()
    y = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()
    class_label = models.CharField(max_length=255)
    confidence = models.FloatField()

    def __str__(self):
        return f"{self.frame} - {self.class_label}"