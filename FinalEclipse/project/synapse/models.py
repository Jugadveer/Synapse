from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)    
    GENDER_CHOICES = (
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other'),
    )
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age = models.IntegerField()
    def __str__(self):
        return self.user.username
    


from django.db import models
from django.contrib.auth.models import User
class ScanResult(models.Model):
    RISK_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    scan_type = models.CharField(max_length=10)  # MRI / AUDIO
    result = models.CharField(max_length=100)
    confidence = models.FloatField()
    risk_level = models.CharField(max_length=10, choices=RISK_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.user.username} - {self.scan_type} - {self.risk_level}"